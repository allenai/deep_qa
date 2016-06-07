package org.allenai.semparse.parse

import com.mattg.util.JsonHelper

import org.json4s._

// TODO(matt): This class is getting pretty big.  Would it make sense to split this into several
// classes, each of which handles specific cases?  Like, a VerbLogicGenerator?  That would also
// allow for easier customization of particular structures.
class LogicalFormGenerator(params: JValue) extends Serializable {

  val validParams = Seq("nested", "split trees")
  JsonHelper.ensureNoExtras(params, "LogicalFormGenerator", validParams)

  val nestLogicalForms = JsonHelper.extractWithDefault(params, "nested", false)
  val shouldSplitTrees = JsonHelper.extractWithDefault(params, "split trees", true)

  def getLogicalForm(tree: DependencyTree): Option[Logic] = {
    val splitTrees = if (shouldSplitTrees) {
      transformers.SplitConjunctions.transform(tree).flatMap(t => {
        transformers.SplitAppositives.transform(t)
      })
    } else {
      Set(tree)
    }
    val transformedTrees = splitTrees.map(defaultTransformations)
    val foundStatements = transformedTrees.flatMap(t => {
      _getLogicForNode(t)
    })
    if (foundStatements.isEmpty) None else Some(Conjunction(foundStatements).flatten)
  }

  def defaultTransformations(tree: DependencyTree): DependencyTree = {
    val withoutDeterminers = transformers.RemoveDeterminers.transform(tree)
    val withoutSuperlatives = transformers.RemoveSuperlatives.transform(withoutDeterminers)
    val passiveUndone = transformers.UndoPassivization.transform(withoutSuperlatives)
    val particlesCombined = transformers.CombineParticles.transform(passiveUndone)
    val ccsRemoved = transformers.RemoveBareCCs.transform(particlesCombined)
    val auxesRemoved = transformers.RemoveAuxiliaries.transform(ccsRemoved)
    auxesRemoved
  }

  def _getLogicForNode(tree: DependencyTree): Option[Logic] = {
    val logicForNode = if (isExistential(tree)) {
      getLogicForExistential(tree)
    } else if (tree.isVerb) {
      getLogicForVerb(tree)
    } else if (isCopula(tree)) {
      getLogicForCopula(tree)
    } else {
      getLogicForOther(tree)
    }
    val childLogic = if (nestLogicalForms) Set() else tree.children.map(_._1).flatMap(_getLogicForNode)
    val combined = logicForNode.toSet ++ childLogic
    if (combined.isEmpty) None else Some(Conjunction(combined))
  }

  /**
   * Predicates are the main logic item we have.  This actually constructs Predicate objects.
   * There shouldn't be any other place in the code where we construct a Predicate, besides these
   * two methods.
   *
   * The reason we put it here is that we want to have a single switch for how we treat these
   * things.  Either we can have a bunch of conjoined simplifications, or we can have nested
   * predicates.
   */
  def getLogicForBinaryPredicate(
    predicate: String,
    arg1: DependencyTree,
    arg2: DependencyTree
  ): Option[Logic] = {
    if (nestLogicalForms) {
      val arg1Logic = _getLogicForNode(arg1)
      val arg2Logic = _getLogicForNode(arg2)
      arg1Logic.flatMap(arg1 => arg2Logic.map(arg2 => Predicate(predicate, Seq(arg1, arg2))))
    } else {
      val predicates: Set[Logic] = for (simplifiedArg1 <- arg1.simplifications;
           simplifiedArg2 <- arg2.simplifications;
           arg1 = simplifiedArg1.lemmaYield;
           arg2 = simplifiedArg2.lemmaYield) yield Predicate(predicate, Seq(Atom(arg1), Atom(arg2)))
      Some(Conjunction(predicates))
    }
  }

  def getLogicForUnaryPredicate(
    predicate: String,
    arg: DependencyTree
  ): Option[Logic] = {
    if (nestLogicalForms) {
      val argLogic = _getLogicForNode(arg)
      argLogic.map(arg => Predicate(predicate, Seq(arg)))
    } else {
      val predicates: Set[Logic] = for (simplifiedArg <- arg.simplifications;
           arg = simplifiedArg.lemmaYield) yield Predicate(predicate, Seq(Atom(arg)))
      Some(Conjunction(predicates))
    }
  }

  def isCopula(tree: DependencyTree): Boolean = {
    tree.getChildWithLabel("cop") match {
      case None => return false
      case Some(_) => return true
    }
  }

  def isExistential(tree: DependencyTree): Boolean = {
    if (tree.token.lemma != "be") return false
    tree.getChildWithLabel("expl") match {
      case None => return false
      case Some(_) => return true
    }
  }

  def isAdjectiveLike(child: (DependencyTree, String)): Boolean = {
    val label = child._2
    label == "amod" || label == "nn" || label == "num"
  }

  def isReducedRelative(tree: DependencyTree): Boolean = {
    tree.children.exists(_._2 == "vmod")
  }

  /**
   * The "other" category is where we handle things that aren't verbs or other special
   * constructions.  Mostly, this will be noun phrases; if it's not a noun phrase, we will probably
   * return None from this method.
   *
   * Our plan for handling noun phrases is to take one dependency off at a time, process it into a
   * logical form, then recurse.  The recursion happens in one of two ways:
   *   1. If we're using nested logical forms, the getLogic* methods are indirectly recursive.
   *   2. If we're not using nested logical forms, getLogicForOther does the recursion, using the
   *      subtree we return from this method.
   *
   * The order in which we handle the modifiers in this method is important for nested logical
   * forms - the things that show up first here will be the outer-most predicates in the logical
   * form.
   */
  def processOneChildForOther(tree: DependencyTree): (Option[Logic], Option[DependencyTree]) = {
    if (tree.children.exists(_._2 == "rcmod")) {
      // These are relative clauses (e.g., "genetic material that is called DNA")
      val child = tree.children.find(_._2 == "rcmod").map(_._1).get
      val npWithoutRelative = transformers.removeTree(tree, child)
      val logic = child.children.filter(_._1.token.posTag == "WDT").headOption match {
        case None => {
          //System.err.println("Error processing relative clause for tree:")
          //tree.print()
          None
        }
        case Some(relativizer) => {
          val relativeClause = transformers.replaceTree(child, relativizer._1, npWithoutRelative)
          getLogicForVerb(relativeClause)
        }
      }
      (logic, Some(npWithoutRelative))
    } else if (tree.children.exists(_._2 == "vmod")) {
      // These are reduced relative clauses (e.g., "genetic material called DNA")
      val child = tree.children.find(_._2 == "vmod").map(_._1).get
      val npWithoutRelative = transformers.removeTree(tree, child)
      val logic = child.getChildWithLabel("dobj") match {
        case None => None
        case Some(dobj) => {
          val verb = transformers.removeTree(child, dobj)
          val verbWithSubject = transformers.addChild(verb, npWithoutRelative, "nsubj")
          val verbWithObject = transformers.addChild(verbWithSubject, dobj, "dobj")
          getLogicForVerb(verbWithObject)
        }
      }
      (logic, Some(npWithoutRelative))
    } else if (tree.children.exists(_._2.startsWith("prep_"))) {
      // Prepositional attachments
      val child = tree.children.find(_._2.startsWith("prep_")).get
      val childTree = child._1
      val label = child._2
      val prep = label.replace("prep_", "")
      val withoutPrep = transformers.removeTree(tree, childTree)
      val logic = getLogicForBinaryPredicate(prep, withoutPrep, childTree)
      (logic, Some(withoutPrep))
    } else if (tree.children.exists(_._2 == "poss")) {
      // Possessives
      val child = tree.children.find(_._2 == "poss").map(_._1).get
      val token = Token("'s", "V", "'s", 0)
      val arg1 = child
      val arg2 = transformers.removeTree(tree, child)
      val logic = getLogicForVerbWithArguments(token, Seq((arg1, "nsubj"), (arg2, "dobj")))
      (logic, Some(arg2))
    } else if (tree.children.exists(isAdjectiveLike)) {
      // Adjectives
      val child = tree.children.find(isAdjectiveLike).map(_._1).get
      val withoutAdjective = transformers.removeTree(tree, child)
      val logic = getLogicForUnaryPredicate(child.token.lemma, withoutAdjective)
      (logic, Some(withoutAdjective))
    } else {
      // We didn't find a modifier to convert into logic.  We'll either return a base Atom, or
      // we'll return nothing.
      if (nestLogicalForms && tree.children.size == 0) {
        // In the nested setting, we need to stop the recursion at an Atom somewhere.  This is where
        // it happens.
        (Some(Atom(tree.token.lemma)), None)
      } else {
        (None, None)
      }
    }
  }

  /**
   * Most frequently, this method will produce logic for noun phrases.  There are a lot of ways
   * that noun phrases can produce logic statements: by having adjective modifiers, prepositional
   * phrase attachments, relative clauses, and so on.  We'll check for them in this method, one by
   * one, handle them, remove them, and recurse.
   */
  def getLogicForOther(tree: DependencyTree): Option[Logic] = {
    val (logic, remainingTree) = processOneChildForOther(tree)
    if (nestLogicalForms) {
      // If we're using nested logical forms, the recursion has already happened when producing the
      // logic that we got, so there is no need to look at the remaining tree.
      logic
    } else {
      // But, if we are not nesting logical forms, the recursion has not happened yet.  We do that
      // here, and add the results to a Conjunction.
      remainingTree match {
        case None => logic
        case Some(tree) => {
          val preds = logic.toSet ++ _getLogicForNode(tree).toSet
          if (preds.isEmpty) None else Some(Conjunction(preds))
        }
      }
    }
  }

  /**
   * Copula sentences are funny cases.  It could just be a noun and an adjective, like "Grass is
   * green".  In these cases, we want to return something like "be(grass, green)".  It could
   * instead be expressing a relationship between two nouns, like "Grass is a kind of plant".  In
   * these cases, we want to grab the relationship and make it the predicate: "kind_of(grass,
   * plant)".  But, this same idea could be expressed as "One kind of plant is grass", and we
   * really want to get the same logical form out for this.  So, there's some complexity in the
   * code to try to figure out if one side or the other of the "is" has a prepositional phrase
   * attachment.
   */
  def getLogicForCopula(tree: DependencyTree): Option[Logic] = {
    tree.getChildWithLabel("nsubj") match {
      case None => None
      case Some(nsubj) => {
        val treeWithoutCopula = transformers.removeChild(tree, "cop")
        val preps = tree.children.filter(_._2.startsWith("prep_"))
        if (preps.size > 0) {
          // We have a prepositional phrase in the usual case ("Grass is a kind of plant").  The
          // Stanford parser will have made "kind" the head word in this case, so we take that as
          // the predicate, and use our standard verb logic to handle this (the verb logic will add
          // the preposition to the predicate name).
          val args = Seq((nsubj, "nsubj")) ++ preps
          getLogicForVerbWithArguments(tree.token, args)
        } else {
          // The usual case ("Grass is a kind of plant") is not present.  So we check for the other
          // case ("One kind of a plant is grass").
          val nsubjPreps = nsubj.children.filter(_._2.startsWith("prep_"))
          if (nsubjPreps.size > 0) {
            // We found it.  So, we do some funny business to switch the argument order, so we get
            // the same predicate out that we would in the usual case.  The token we want to use
            // this time is the nsubj, the tree minus the nsubj is the first argument, and the
            // prepositional phrase attachments are the remaining arguments.
            val withoutNsubj = transformers.removeTree(treeWithoutCopula, nsubj)
            val args = Seq((withoutNsubj, "nsubj")) ++ nsubjPreps
            getLogicForVerbWithArguments(nsubj.token, args)
          } else {
            // No prepositional phrase found, so just use "be" as the predicate.
            val withoutCopAndSubj = transformers.removeTree(treeWithoutCopula, nsubj)
            val args = Seq((withoutCopAndSubj, "nsubj"), (nsubj, "dobj"))
            getLogicForVerbWithArguments(Token("be", "V", "be", 0), args)
          }
        }
      }
    }
  }

  def getLogicForExistential(tree: DependencyTree): Option[Logic] = {
    tree.getChildWithLabel("nsubj") match {
      case None => {
        // This is a nested existential...  Let's just ignore it for now.
        None
      }
      case Some(child) => { getLogicForUnaryPredicate("exists", child) }
    }
  }

  def getLogicForVerb(tree: DependencyTree): Option[Logic] = {
    if (tree.children.exists(c => c._2 == "xcomp" || c._2.startsWith("prepc_"))) {
      return getLogicForControllingVerb(tree)
    }
    val arguments = getVerbArguments(tree)
    getLogicForVerbWithArguments(tree.token, arguments)
  }

  def getLogicForControllingVerb(tree: DependencyTree): Option[Logic] = {
    // Passives _with_ agents are handled in the tree transformations, and most other constructions
    // with "is" or "are" are handled by getLogicForCopula.  Except, in some control sentences
    // (like "Tools are used to measure things"), we actually have two arguments, and one of them
    // is technically the subject of a passive.  We change that passive into the upper argument of
    // a control verb here.
    val rootArguments = getVerbArguments(tree).map(arg => {
      if (arg._2 == "dobj") {
        (arg._1, "upper_dobj")
      } else {
        arg
      }
    }).filterNot(_._2 == "nsubjpass") ++ getPassiveSubject(tree).map(arg => (arg._1, "upper_dobj"))

    // TODO(matt): I should probably handle this as a map, instead of a find, in case we have
    // both...  I'll worry about that when I see it in a sentence, though.
    val controlledVerb = tree.children.find(c => c._2 == "xcomp" || c._2.startsWith("prepc_")).head
    val controlledArguments = getVerbArguments(controlledVerb._1).map(arg => {
      if (arg._2 == "dobj" && controlledVerb._2 == "xcomp") {
        (arg._1, "lower_dobj")
      } else {
        arg
      }
    })
    val arguments = (rootArguments ++ controlledArguments).sortBy(argumentSortKey)
    val combiner = if (controlledVerb._2 == "xcomp") {
      tree.getChildWithLabel("csubj") match {
        case None => "Csubj"
        case Some(_) => "Cobj"
      }
    } else {
      controlledVerb._2.replace("prepc_", "")
    }
    val tokenWithCombiner = tree.token.addPreposition(combiner)
    val combinedToken = tokenWithCombiner.combineWith(controlledVerb._1.token)
    getLogicForVerbWithArguments(combinedToken, arguments, allowEmptySubject=true)
  }

  def getLogicForVerbWithArguments(
    token: Token,
    arguments: Seq[(DependencyTree, String)],
    allowEmptySubject: Boolean = false
  ): Option[Logic] = {
    if (nestLogicalForms) {
      getNestedLogicForVerbArguments(token, arguments, allowEmptySubject)
    } else {
      getCrossProductOfVerbArguments(token, arguments, allowEmptySubject)
    }
  }

  def getNestedLogicForVerbArguments(
    token: Token,
    arguments: Seq[(DependencyTree, String)],
    allowEmptySubject: Boolean
  ): Option[Logic] = {
    // The basic strategy here is to order the arguments in an adjunct hierarchy, create a base
    // predicate with the core arguments, and nest that inside of the next argument, recursively.
    // So, for example, "animals depend on plants for food" would give `for(depend_on(animals,
    // plants), food)`
    val logic = getCrossProductOfVerbArguments(token, arguments.take(2), allowEmptySubject)
    logic match {
      case None => None
      case Some(logic) => {
        var currentLogic = logic
        for (argument <- arguments.drop(2)) {
          val predicate = argument._2.replace("prep_", "")
          val argumentLogic = getLogicalForm(argument._1)
          argumentLogic match {
            case None => {
              // Really not sure what to do here...  Need to see when this happens.  Let's just punt
              // for now, and do nothing.
            }
            case Some(argumentLogic) => {
              currentLogic = Predicate(predicate, Seq(currentLogic, argumentLogic))
            }
          }
        }
        Some(currentLogic)
      }
    }
  }

  def getCrossProductOfVerbArguments(
    token: Token,
    arguments: Seq[(DependencyTree, String)],
    allowEmptySubject: Boolean
  ): Option[Logic] = {
    // If we got here with a verb that has no subject, bail.  This may have happened when checking
    // the children of a control verb, or a few other odd constructions.
    if (!allowEmptySubject && !arguments.exists(_._2.contains("subj"))) return None
    // First we handle the case where we have one argument.  This is an intransitive verb (or a
    // transitive verb in the passive voice with no agent; our tree transformations will handle the
    // case where there is an agent).
    if (arguments.size == 1) {
      val tokenWithArg = modifyPredicateForArg1(token, arguments(0))
      return getLogicForUnaryPredicate(tokenWithArg.lemma, arguments(0)._1)
    }
    // Now the other cases.  We might have more than two arguments, so we'll handle this by taking
    // all pairs of arguments.  The arguments have already been sorted, so the subject is first,
    // then the object (if any), then any prepositional arguments.
    val logic = for (i <- 0 until arguments.size;
                     j <- (i + 1) until arguments.size) yield {
      val arg1 = arguments(i)
      val arg2 = arguments(j)
      val tokenWithArg1 = modifyPredicateForArg1(token, arg1)
      val tokenWithArg2 = modifyPredicateForArg2(tokenWithArg1, arg2)
      getLogicForTransitiveVerb(tokenWithArg2.lemma, Seq(arg1, arg2))
    }
    val preds = logic.flatten.toSet
    if (preds.isEmpty) None else Some(Conjunction(preds))
  }

  def modifyPredicateForArg1(token: Token, argument: (DependencyTree, String)): Token = {
    if (argument._2 == "nsubjpass") {
      // This is a passive verb without an agent.  In this case, we are going to use the participle
      // as the predicate, instead of the lemma, to differentiate it from the active voice.  So we
      // just return a new token with `token.word` copied into `token.lemma`.
      Token(token.word, token.posTag, token.word, token.index)
    } else if (argument._2.startsWith("prep_")) {
      val prep = argument._2.replace("prep_", "")
      token.addPreposition(prep)
    } else if (argument._2.endsWith("dobj")) {
      token.addPreposition("obj")
    } else if (argument._2 == "iobj") {
      token.addPreposition("obj2")
    } else {
      token
    }
  }

  def modifyPredicateForArg2(token: Token, argument: (DependencyTree, String)): Token = {
    if (argument._2.startsWith("prep_")) {
      val prep = argument._2.replace("prep_", "")
      token.addPreposition(prep)
    } else if (argument._2 == "lower_dobj") {
      token.addPreposition("obj")
    } else if (argument._2 == "iobj") {
      token.addPreposition("obj2")
    } else {
      token
    }
  }

  def getLogicForTransitiveVerb(predicate: String, arguments: Seq[(DependencyTree, String)]): Option[Logic] = {
    if (arguments.exists(_._1.isWhPhrase)) return None
    getLogicForBinaryPredicate(predicate, arguments(0)._1, arguments(1)._1)
  }

  def argumentSortKey(arg: (DependencyTree, String)) = {
    val label = arg._2
    val sortIndex = label match {
      case "nsubj" => 1
      case "nsubjpass" => 2  // we transform this if there's an agent, but there might not be one
      case "csubj" => 3
      case "dobj" => 4
      case "upper_dobj" => 5
      case "lower_dobj" => 6
      case "iobj" => 7
      case _ => 8
    }
    (sortIndex, arg._1.tokenStartIndex)
  }

  def getVerbArguments(tree: DependencyTree) = {
    val arguments = tree.children.filter(c => {
      val label = c._2
      label.contains("subj") || label == "dobj" || label == "iobj" || label.startsWith("prep_")
    })
    arguments.sortBy(argumentSortKey)
  }

  def getPassiveSubject(tree: DependencyTree) = {
    tree.children.filter(_._2 == "nsubjpass")
  }
}

package org.allenai.semparse.parse

import com.mattg.util.JsonHelper

import org.json4s._

class LogicalFormGenerator(params: JValue) {

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

  def isReducedRelative(tree: DependencyTree): Boolean = {
    tree.children.exists(_._2 == "vmod")
  }

  def getLogicForReducedRelative(tree: DependencyTree): Set[Logic] = {
    tree.children.filter(_._2 == "vmod").map(_._1).flatMap(child => {
      val npWithoutRelative = transformers.removeTree(tree, child)
      child.getChildWithLabel("dobj") match {
        case None => Set[Predicate]()
        case Some(dobj) => {
          val verb = transformers.removeTree(child, dobj)
          val verbWithSubject = transformers.addChild(verb, npWithoutRelative, "nsubj")
          val verbWithObject = transformers.addChild(verbWithSubject, dobj, "dobj")
          getLogicForVerb(verbWithObject)
        }
      }
    }).toSet
  }

  /**
   * Most frequently, this method will produce logic for nouns.  There are a lot of ways that nouns
   * can produce logic statements: by having adjective modifiers, prepositional phrase attachments,
   * relative clauses, and so on.  We'll check for them in this method, and merge them all at the
   * end.  The merging is a little complicated for nested predicates, though.
   */
  def getLogicForOther(tree: DependencyTree): Option[Logic] = {
    val wordPredicates = if (nestLogicalForms && tree.children.size == 0) {
      // In the nested setting, we need to stop the recursion at an Atom somewhere.  This is where
      // it happens.
      Set(Atom(tree.token.lemma))
    } else {
      Set()
    }

    val adjectivePredicates = tree.children.filter(c => c._2 == "amod" || c._2 == "nn").map(_._1).map(child => {
      Predicate(child.token.lemma, Seq(Atom(tree.token.lemma)))
    }).toSet

    val reducedRelativeClausePredicates = getLogicForReducedRelative(tree)

    val relativeClausePredicates = tree.children.filter(_._2 == "rcmod").map(_._1).flatMap(child => {
      val npWithoutRelative = transformers.removeTree(tree, child)
      child.children.filter(_._1.token.posTag == "WDT").headOption match {
        case None => {
          System.err.println("Error processing relative clause for tree:")
          tree.print()
          Seq()
        }
        case Some(relativizer) => {
          val relativeClause = transformers.replaceTree(child, relativizer._1, npWithoutRelative)
          getLogicForVerb(relativeClause)
        }
      }
    })

    val prepositionPredicates = tree.children.filter(_._2.startsWith("prep_")).flatMap(child => {
      val childTree = child._1
      val label = child._2
      val prep = label.replace("prep_", "")
      val withoutPrep = transformers.removeTree(tree, childTree)
      getLogicForBinaryPredicate(prep, withoutPrep, childTree)
    })

    // Finally, we get to the merge step.
    val preds = if (nestLogicalForms) {
      // This is complicated if we're nesting logical forms, because some of these end up being
      // dupcliates.  We want to set an order of precedence, and only return the top-most level,
      // because everything else will have already been generated as part of that logical form.
      // For example, in the phrase "genetic material called dna", there's a reduced relative
      // clause that generates the predicate `call(genetic(material), dna)`.  We don't want to also
      // produce `genetic(material)` from the adjective predicates above, because that duplicates
      // what we've already included with the nesting.
      if (!reducedRelativeClausePredicates.isEmpty) {
        reducedRelativeClausePredicates
      } else {
        wordPredicates ++ adjectivePredicates ++ prepositionPredicates ++ relativeClausePredicates
      }
    } else {
      // If we aren't nesting logical forms, this is trivial - we just add them all into a set.
      wordPredicates ++
      adjectivePredicates ++
      reducedRelativeClausePredicates ++
      prepositionPredicates ++
      relativeClausePredicates
    }
    if (preds.isEmpty) None else Some(Conjunction(preds))
  }

  def getLogicForCopula(tree: DependencyTree): Option[Logic] = {
    tree.getChildWithLabel("nsubj") match {
      case None => None
      case Some(nsubj) => {
        val treeWithoutCopula = transformers.removeChild(tree, "cop")
        val preps = tree.children.filter(_._2.startsWith("prep_"))
        if (preps.size == 0) {
          val withoutCopAndSubj = transformers.removeTree(treeWithoutCopula, nsubj)
          val args = Seq((withoutCopAndSubj, "nsubj"), (nsubj, "dobj"))
          getLogicForVerbWithArguments(Token("be", "V", "be", 0), args)
        } else {
          val args = Seq((nsubj, "nsubj")) ++ preps
          getLogicForVerbWithArguments(tree.token, args)
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
    // Control verbs are the only place we're going to try to handle passives without agents.
    // Passives _with_ agents are handled in the tree transformations, and most other constructions
    // with "is" or "are" are handled by getLogicForCopula.  Except, in some control sentences, we
    // actually have two arguments, and one of them is technically the subject of a passive.  We
    // change that passive into the upper argument of a control verb here.
    val rootArguments = getVerbArguments(tree).map(arg => {
      if (arg._2 == "dobj") {
        (arg._1, "upper_dobj")
      } else {
        arg
      }
    }) ++ getPassiveSubject(tree).map(arg => (arg._1, "upper_dobj"))

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
    getLogicForVerbWithArguments(combinedToken, arguments)
  }

  def getLogicForVerbWithArguments(token: Token, arguments: Seq[(DependencyTree, String)]): Option[Logic] = {
    if (nestLogicalForms) {
      getNestedLogicForVerbArguments(token, arguments)
    } else {
      getCrossProductOfVerbArguments(token, arguments)
    }
  }

  def getNestedLogicForVerbArguments(token: Token, arguments: Seq[(DependencyTree, String)]): Option[Logic] = {
    // The basic strategy here is to order the arguments in an adjunct hierarchy, create a base
    // predicate with the core arguments, and nest that inside of the next argument, recursively.
    // So, for example, "animals depend on plants for food" would give `for(depend_on(animals,
    // plants), food)`
    val logic = getCrossProductOfVerbArguments(token, arguments.take(2))
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

  def getCrossProductOfVerbArguments(token: Token, arguments: Seq[(DependencyTree, String)]): Option[Logic] = {
    // We might have more than two arguments, so we'll handle this by taking all pairs of
    // arguments.  The arguments have already been sorted, so the subject is first, then the object
    // (if any), then any prepositional arguments.
    val logic = for (i <- 0 until arguments.size;
                     j <- (i + 1) until arguments.size) yield {
      val arg1 = arguments(i)
      val arg2 = arguments(j)
      val tokenWithArg1Prep = if (arg1._2.startsWith("prep_")) {
        val prep = arg1._2.replace("prep_", "")
        token.addPreposition(prep)
      } else if (arg1._2.endsWith("dobj")) {
        token.addPreposition("obj")
      } else if (arg1._2 == "iobj") {
        token.addPreposition("obj2")
      } else {
        token
      }
      val tokenWithArg2Prep = if (arg2._2.startsWith("prep_")) {
        val prep = arg2._2.replace("prep_", "")
        tokenWithArg1Prep.addPreposition(prep)
      } else if (arg2._2 == "lower_dobj") {
        tokenWithArg1Prep.addPreposition("obj")
      } else if (arg2._2 == "iobj") {
        tokenWithArg1Prep.addPreposition("obj2")
      } else {
        tokenWithArg1Prep
      }
      getLogicForTransitiveVerb(tokenWithArg2Prep, Seq(arg1, arg2))
    }
    val preds = logic.flatten.toSet
    if (preds.isEmpty) None else Some(Conjunction(preds))
  }

  def getLogicForTransitiveVerb(token: Token, arguments: Seq[(DependencyTree, String)]): Option[Logic] = {
    if (arguments.exists(_._1.isWhPhrase)) return None
    getLogicForBinaryPredicate(token.lemma, arguments(0)._1, arguments(1)._1)
  }

  def argumentSortKey(arg: (DependencyTree, String)) = {
    val label = arg._2
    val sortIndex = label match {
      case "nsubj" => 1
      case "csubj" => 2
      case "dobj" => 3
      case "upper_dobj" => 4
      case "lower_dobj" => 5
      case "iobj" => 6
      case _ => 7
    }
    (sortIndex, arg._1.tokenStartIndex)
  }

  def getVerbArguments(tree: DependencyTree) = {
    val arguments = tree.children.filter(c => {
      val label = c._2
      label.endsWith("subj") || label == "dobj" || label == "iobj" || label.startsWith("prep_")
    })
    arguments.sortBy(argumentSortKey)
  }

  def getPassiveSubject(tree: DependencyTree) = {
    tree.children.filter(_._2 == "nsubjpass")
  }

}

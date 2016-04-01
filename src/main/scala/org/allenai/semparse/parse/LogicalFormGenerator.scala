package org.allenai.semparse.parse

case class Predicate(predicate: String, arguments: Seq[String]) {
  override def toString(): String = {
    val argString = arguments.mkString(", ")
    s"$predicate(${arguments.mkString(", ")})"
  }
}

object LogicalFormGenerator {

  // TODO(matt): At some point I'm going to have to make this return a Logic argument, instead of
  // just a set of Predicates, so that I can handle negation, and disjunction, and other things.
  // But this is good enough for now.
  def getLogicalForm(tree: DependencyTree): Set[Predicate] = {
    val splitTrees = transformers.SplitConjunctions.transform(tree).flatMap(t => {
      transformers.SplitAppositives.transform(t)
    })
    val transformedTrees = splitTrees.map(defaultTransformations)
    transformedTrees.flatMap(t => {
      _getLogicForNode(t)
    })
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

  def _getLogicForNode(tree: DependencyTree): Set[Predicate] = {
    val logicForNode = if (isExistential(tree)) {
      getLogicForExistential(tree)
    } else if (tree.isVerb) {
      getLogicForVerb(tree)
    } else if (isCopula(tree)) {
      getLogicForCopula(tree)
    } else {
      getLogicForOther(tree)
    }
    logicForNode ++ tree.children.map(_._1).flatMap(_getLogicForNode)
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

  def getLogicForOther(tree: DependencyTree): Set[Predicate] = {
    val adjectivePredicates = tree.children.filter(c => c._2 == "amod" || c._2 == "nn").map(_._1).map(child => {
      Predicate(child.token.lemma, Seq(tree.token.lemma))
    }).toSet
    val relativeClausePredicates = tree.children.filter(_._2 == "vmod").map(_._1).flatMap(child => {
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
    })
    val prepositionPredicates = tree.children.filter(_._2.startsWith("prep_")).flatMap(child => {
      val childTree = child._1
      val label = child._2
      val prep = label.replace("prep_", "")
      val withoutPrep = transformers.removeTree(tree, childTree)
      for (simplifiedTree <- withoutPrep.simplifications;
           simplifiedChild <- childTree.simplifications;
           subj = simplifiedTree.lemmaYield;
           arg = simplifiedChild.lemmaYield) yield Predicate(prep, Seq(subj, arg))
    })
    adjectivePredicates ++ relativeClausePredicates ++ prepositionPredicates
  }

  def getLogicForCopula(tree: DependencyTree): Set[Predicate] = {
    tree.getChildWithLabel("nsubj") match {
      case None => Set[Predicate]()
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

  def getLogicForExistential(tree: DependencyTree): Set[Predicate] = {
    tree.getChildWithLabel("nsubj") match {
      case None => {
        // This is a nested existential...  Let's just ignore it for now.
        Set()
      }
      case Some(child) => {
        for (simplified <- child.simplifications)
          yield Predicate("exists", Seq(simplified.lemmaYield))
      }
    }
  }

  def getLogicForVerb(tree: DependencyTree): Set[Predicate] = {
    if (tree.children.exists(c => c._2 == "xcomp" || c._2.startsWith("prepc_"))) {
      return getLogicForControllingVerb(tree)
    }
    val arguments = getVerbArguments(tree)
    getLogicForVerbWithArguments(tree.token, arguments)
  }

  def getLogicForControllingVerb(tree: DependencyTree): Set[Predicate] = {
    val rootArguments = getVerbArguments(tree)
    // TODO(matt): I should probably handle this as a map, instead of a find, in case we have
    // both...  I'll worry about that when I see it in a sentence, though.
    val controlledVerb = tree.children.find(c => c._2 == "xcomp" || c._2.startsWith("prepc_")).head
    val controlledArguments = getVerbArguments(controlledVerb._1)
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

  def getLogicForVerbWithArguments(token: Token, arguments: Seq[(DependencyTree, String)]): Set[Predicate] = {
    // We have more than two arguments, so we'll handle this by taking all pairs of arguments.
    // The arguments have already been sorted, so the subject is first, then the object (if any),
    // then any prepositional arguments.
    val logic = for (i <- 0 until arguments.size;
                     j <- (i + 1) until arguments.size) yield {
      val arg1 = arguments(i)
      val arg2 = arguments(j)
      val tokenWithArg1Prep = if (arg1._2.startsWith("prep_")) {
        val prep = arg1._2.replace("prep_", "")
        token.addPreposition(prep)
      } else if (arg1._2 == "dobj") {
        token.addPreposition("obj")
      } else if (arg1._2 == "iobj") {
        token.addPreposition("obj2")
      } else {
        token
      }
      val tokenWithArg2Prep = if (arg2._2.startsWith("prep_")) {
        val prep = arg2._2.replace("prep_", "")
        tokenWithArg1Prep.addPreposition(prep)
      } else if (arg2._2 == "iobj") {
        tokenWithArg1Prep.addPreposition("obj2")
      } else {
        tokenWithArg1Prep
      }
      getLogicForTransitiveVerb(tokenWithArg2Prep, Seq(arg1, arg2))
    }
    logic.flatten.toSet
  }

  def getLogicForTransitiveVerb(token: Token, arguments: Seq[(DependencyTree, String)]): Set[Predicate] = {
    val logic =
      for (subjTree <- arguments(0)._1.simplifications;
           subj = subjTree.lemmaYield;
           objTree <- arguments(1)._1.simplifications;
           obj = objTree.lemmaYield) yield Predicate(token.lemma, Seq(subj, obj))
    logic.toSet
  }

  def argumentSortKey(arg: (DependencyTree, String)) = {
    val label = arg._2
    val sortIndex = label match {
      case "nsubj" => 1
      case "csubj" => 2
      case "dobj" => 3
      case "iobj" => 4
      case _ => 5
    }
    (sortIndex, label)
  }

  def getVerbArguments(tree: DependencyTree) = {
    val arguments = tree.children.filter(c => {
      val label = c._2
      label.endsWith("subj") || label == "dobj" || label == "iobj" || label.startsWith("prep_")
    })
    arguments.sortBy(argumentSortKey)
  }

}

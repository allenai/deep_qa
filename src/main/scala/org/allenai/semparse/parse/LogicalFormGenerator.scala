package org.allenai.semparse.parse

case class Predicate(predicate: String, arguments: Seq[String]) {
  override def toString(): String = {
    val argString = arguments.mkString(", ")
    s"$predicate(${arguments.mkString(", ")})"
  }
}

object LogicalFormGenerator {
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
    val treeWithoutCopula = transformers.removeChild(tree, "cop")
    val preps = tree.children.filter(_._2.startsWith("prep_"))
    if (preps.size == 0) {
      tree.children.find(_._2 == "nsubj") match {
        case None => Set[Predicate]()
        case Some(nsubjWithLabel) => {
          val nsubj = nsubjWithLabel._1
          val withoutCopAndSubj = transformers.removeTree(treeWithoutCopula, nsubj)
          for (simplifiedTree <- withoutCopAndSubj.simplifications;
               simplifiedNsubj <- nsubj.simplifications;
               pred = simplifiedTree.lemmaYield;
               subj = simplifiedNsubj.lemmaYield) yield Predicate("be", Seq(pred, subj))
        }
      }
    } else if (preps.size == 1) {
      tree.children.find(_._2 == "nsubj") match {
        case None => Set[Predicate]()
        case Some(nsubjWithLabel) => {
          val prepChild = preps.head
          val prepTree = prepChild._1
          val prep = prepChild._2.replace("prep_", "")
          val predicate = tree.token.addPreposition(prep).lemma
          val nsubj = nsubjWithLabel._1
          for (simplifiedNsubj <- nsubj.simplifications;
               simplifiedPrep <- prepTree.simplifications;
               subj = simplifiedNsubj.lemmaYield;
               p = simplifiedPrep.lemmaYield) yield Predicate(predicate, Seq(subj, p))
        }
      }
    } else {
      // We'll punt in this case...
      System.err.println(s"Not sure how to handle case with two prepositions: ${tree}")
      Set[Predicate]()
    }
  }

  def getLogicForExistential(tree: DependencyTree): Set[Predicate] = {
    val child = tree.getChildWithLabel("nsubj").get
    for (simplified <- child.simplifications) yield Predicate("exists", Seq(simplified.lemmaYield))
  }

  def getLogicForVerb(tree: DependencyTree): Set[Predicate] = {
    if (tree.children.exists(c => c._2 == "xcomp" || c._2.startsWith("prepc_"))) {
      return getLogicForControllingVerb(tree)
    }
    val arguments = getVerbArguments(tree)
    getLogicForVerbWithArguments(tree.token, arguments)
  }

  def getLogicForVerbWithArguments(token: Token, arguments: Seq[(DependencyTree, String)]): Set[Predicate] = {
    arguments.size match {
      case 0 => Set()
      case 1 => getLogicForOneArgVerb(token, arguments)
      case 2 => getLogicForTwoArgVerb(token, arguments)
      case _ => getLogicForMoreThanTwoArgVerb(token, arguments)
    }
  }

  def getLogicForOneArgVerb(token: Token, arguments: Seq[(DependencyTree, String)]): Set[Predicate] = {
    if (arguments(0)._2 == "nsubj") {
      val subject = arguments(0)._1
      subject.simplifications.map(s => Predicate(token.lemma, Seq(s.lemmaYield)))
    } else if (arguments(0)._2 == "dobj") {
      // The plan is that we've dealt with these cases higher up in the tree, so we don't need to
      // deal with them here, and we'll just ignore it.  This means that we don't get any logic for
      // imperative sentences, but I think that's fine for science sentences.
      Set()
    } else {
      Set()
    }
  }

  def getLogicForTwoArgVerb(token: Token, arguments: Seq[(DependencyTree, String)]): Set[Predicate] = {
    if (arguments(0)._2.endsWith("subj") && arguments(1)._2 == "dobj") {
      getLogicForTransitiveVerb(token, arguments)
    } else if (arguments(0)._2.endsWith("subj") && arguments(1)._2.startsWith("prep_")) {
      val prep = arguments(1)._2.replace("prep_", "")
      val newToken = token.addPreposition(prep)
      getLogicForTransitiveVerb(newToken, arguments)
    } else if (arguments(0)._2 == "dobj" && arguments(1)._2.startsWith("prep_")) {
      val prep = "obj_" + arguments(1)._2.replace("prep_", "")
      val newToken = token.addPreposition(prep)
      getLogicForTransitiveVerb(newToken, arguments)
    } else if (arguments(0)._2 == "nsubj" && arguments(1)._2.startsWith("nsubj")) {
      // We're just going to skip this...  It's pretty much just due to errors, like "When I
      // first I read about ..."
      Set()
    } else {
      System.err.println(s"unhandled 2-argument verb case: ${token}, ${arguments}")
      Set()
    }
  }

  def getLogicForMoreThanTwoArgVerb(token: Token, arguments: Seq[(DependencyTree, String)]): Set[Predicate] = {
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

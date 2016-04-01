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
      _getLogicForNode(t) ++ t.children.map(_._1).flatMap(_getLogicForNode)
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
    if (isExistential(tree)) {
      getLogicForExistential(tree)
    } else if (tree.isVerb) {
      getLogicForVerb(tree)
    } else if (isCopula(tree)) {
      getLogicForCopula(tree)
    } else {
      getLogicForOther(tree)
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
    if (arguments.size == 1) {
      getLogicForOneArgVerb(tree, arguments)
    } else if (arguments.size == 2) {
      getLogicForTwoArgVerb(tree, arguments)
    } else {
      getLogicForMoreThanTwoArgVerb(tree, arguments)
    }
  }

  def getLogicForOneArgVerb(tree: DependencyTree, arguments: Seq[(DependencyTree, String)]): Set[Predicate] = {
    if (arguments(0)._2 == "nsubj") {
      val subject = arguments(0)._1
      subject.simplifications.map(s => Predicate(tree.token.lemma, Seq(s.lemmaYield)))
    } else if (arguments(0)._2 == "dobj") {
      val obj = arguments(0)._1
      obj.simplifications.map(s => Predicate(tree.token.lemma + "_obj", Seq(s.lemmaYield)))
    } else {
      Set()
    }
  }

  def getLogicForTwoArgVerb(tree: DependencyTree, arguments: Seq[(DependencyTree, String)]): Set[Predicate] = {
    if (arguments(0)._2 == "nsubj" && arguments(1)._2 == "dobj") {
      getLogicForTransitiveVerb(tree, arguments)
    } else if (arguments(0)._2 == "nsubj" && arguments(1)._2.startsWith("prep_")) {
      val prep = arguments(1)._2.replace("prep_", "")
      val newToken = tree.token.addPreposition(prep)
      getLogicForTransitiveVerb(DependencyTree(newToken, tree.children), arguments)
    } else if (arguments(0)._2 == "dobj" && arguments(1)._2.startsWith("prep_")) {
      val prep = "obj_" + arguments(1)._2.replace("prep_", "")
      val newToken = tree.token.addPreposition(prep)
      getLogicForTransitiveVerb(DependencyTree(newToken, tree.children), arguments)
    } else if (arguments(0)._2 == "nsubj" && arguments(1)._2.startsWith("nsubj")) {
      // We're just going to skip this...  It's pretty much just due to errors, like "When I
      // first I read about ..."
      Set()
    } else {
      System.err.println(s"unhandled 2-argument verb case: ${tree}")
      Set()
    }
  }

  def getLogicForMoreThanTwoArgVerb(tree: DependencyTree, arguments: Seq[(DependencyTree, String)]): Set[Predicate] = {
    // We have more than two arguments, so we'll handle this by taking all pairs of arguments.
    // The arguments have already been sorted, so the subject is first, then the object (if any),
    // then any prepositional arguments.
    val logic = for (i <- 0 until arguments.size;
                     j <- (i + 1) until arguments.size) yield {
      val arg1 = arguments(i)
      val arg2 = arguments(j)
      val tokenWithArg1Prep = if (arg1._2.startsWith("prep_")) {
        val prep = arg1._2.replace("prep_", "")
        tree.token.addPreposition(prep)
      } else if (arg1._2 == "dobj") {
        tree.token.addPreposition("obj")
      } else if (arg1._2 == "iobj") {
        tree.token.addPreposition("obj2")
      } else {
        tree.token
      }
      val tokenWithArg2Prep = if (arg2._2.startsWith("prep_")) {
        val prep = arg2._2.replace("prep_", "")
        tokenWithArg1Prep.addPreposition(prep)
      } else if (arg2._2 == "iobj") {
        tokenWithArg1Prep.addPreposition("obj2")
      } else {
        tokenWithArg1Prep
      }
      val newTree = DependencyTree(tokenWithArg2Prep, tree.children)
      getLogicForTransitiveVerb(newTree, Seq(arg1, arg2))
    }
    logic.flatten.toSet
  }

  def getLogicForTransitiveVerb(
    tree: DependencyTree,
    arguments: Seq[(DependencyTree, String)]
  ): Set[Predicate] = {
    val lemma = tree.token.lemma
    val logic =
      for (subjTree <- arguments(0)._1.simplifications;
           subj = subjTree.lemmaYield;
           objTree <- arguments(1)._1.simplifications;
           obj = objTree.lemmaYield) yield Predicate(lemma, Seq(subj, obj))
    logic.toSet
  }

  def getLogicForControllingVerb(tree: DependencyTree): Set[Predicate] = {
    tree.getChildWithLabel("dobj") match {
      case None => getLogicForSubjectControl(tree)
      case Some(_) => getLogicForObjectControl(tree)
    }
  }

  def getLogicForSubjectControl(tree: DependencyTree): Set[Predicate] = {
    Set()
  }

  def getLogicForObjectControl(tree: DependencyTree): Set[Predicate] = {
    Set()
  }

  def getVerbArguments(tree: DependencyTree) = {
    val arguments = tree.children.filter(c => {
      val label = c._2
      label == "nsubj" || label == "dobj" || label == "iobj" || label.startsWith("prep_")
    })
    arguments.sortBy(arg => {
      val label = arg._2
      val sortIndex = label match {
        case "nsubj" => 1
        case "dobj" => 2
        case "iobj" => 3
        case _ => 4
      }
      (sortIndex, label)
    })
  }

}

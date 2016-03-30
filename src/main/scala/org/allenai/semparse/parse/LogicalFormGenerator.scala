package org.allenai.semparse.parse

case class Predicate(predicate: String, arguments: Seq[String])

object LogicalFormGenerator {
  def getLogicalForm(tree: DependencyTree): Set[Predicate] = {
    val splitTrees = transformers.SplitConjunctions.transform(tree)
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
    particlesCombined
  }

  def _getLogicForNode(tree: DependencyTree): Set[Predicate] = {
    if (tree.isVerb) {
      getLogicForVerb(tree)
    } else {
      getLogicForOther(tree)
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
      for (simplifiedTree <- childTree.simplifications;
           arg = simplifiedTree.lemmaYield) yield Predicate(prep, Seq(tree.token.lemma, arg))
    })
    adjectivePredicates ++ relativeClausePredicates ++ prepositionPredicates
  }

  def getLogicForVerb(tree: DependencyTree): Set[Predicate] = {
    val verbLemma = tree.token.lemma
    val arguments = getVerbArguments(tree)
    if (arguments.size == 1) {
      if (arguments(0)._2 != "nsubj") {
        tree.print()
        throw new RuntimeException(s"One argument verb with no subject: ${arguments(0)}")
      }
      val subject = arguments(0)._1
      subject.simplifications.map(s => Predicate(verbLemma, Seq(s.lemmaYield)))
    } else if (arguments.size == 2) {
      if (arguments(0)._2 == "nsubj" && arguments(1)._2 == "dobj") {
        getLogicForTransitiveVerb(tree, arguments)
      } else if (arguments(0)._2 == "nsubj" && arguments(1)._2.startsWith("prep_")) {
        val prep = arguments(1)._2.replace("prep_", "")
        val newToken = tree.token.addPreposition(prep)
        getLogicForTransitiveVerb(DependencyTree(newToken, tree.children), arguments)
      } else {
        tree.print()
        throw new RuntimeException("unhandled 2-argument verb case")
      }
    } else {
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
        } else {
          tree.token
        }
        val tokenWithArg2Prep = if (arg2._2.startsWith("prep_")) {
          val prep = arg2._2.replace("prep_", "")
          tokenWithArg1Prep.addPreposition(prep)
        } else {
          tokenWithArg1Prep
        }
        val newTree = DependencyTree(tokenWithArg2Prep, tree.children)
        getLogicForTransitiveVerb(newTree, Seq(arg1, arg2))
      }
      logic.flatten.toSet
    }
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

  def getVerbArguments(tree: DependencyTree) = {
    val arguments = tree.children.filter(c => {
      val label = c._2
      label == "nsubj" || label == "dobj" || label.startsWith("prep")
    })
    arguments.sortBy(arg => {
      val label = arg._2
      val sortIndex = label match {
        case "nsubj" => 1
        case "dobj" => 2
        case _ => 3
      }
      (sortIndex, label)
    })
  }

}

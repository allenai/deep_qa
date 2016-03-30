package org.allenai.semparse.parse

case class Predicate(predicate: String, arguments: Seq[String])

object LogicalFormGenerator {
  def getLogicalForm(tree: DependencyTree): Set[Predicate] = {
    val transformedTree = defaultTransformations(tree)
    _getLogicForNode(transformedTree) ++ transformedTree.children.map(_._1).flatMap(_getLogicForNode)
  }

  def defaultTransformations(tree: DependencyTree): DependencyTree = {
    val withoutDeterminers = transformers.RemoveDeterminers.transform(tree)
    val passiveUndone = transformers.UndoPassivization.transform(withoutDeterminers)
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
    val adjectivePredicates = tree.children.filter(_._2 == "amod").map(_._1).map(child => {
      Predicate(child.token.lemma, Seq(tree.token.word))
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
           arg = simplifiedTree._yield) yield Predicate(prep, Seq(tree.token.word, arg))
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
      subject.simplifications.map(s => Predicate(verbLemma, Seq(s._yield)))
    } else if (arguments.size == 2) {
      if (arguments(0)._2 == "nsubj" && arguments(1)._2 == "dobj") {
        getLogicForTransitiveVerb(tree, arguments)
      } else if (arguments(0)._2 == "nsubj" && arguments(1)._2.startsWith("prep_")) {
        val prep = arguments(1)._2.replace("prep_", "")
        val newWord = tree.token.word + "_" + prep
        val newLemma = tree.token.lemma + "_" + prep
        val newToken = Token(newWord, tree.token.posTag, newLemma, tree.token.index)
        getLogicForTransitiveVerb(DependencyTree(newToken, tree.children), arguments)
      } else {
        tree.print()
        throw new RuntimeException("unhandled 2-argument verb case")
      }
    } else {
      tree.print()
      throw new RuntimeException("verbs with more than two arguments not yet handled")
    }
  }

  def getLogicForTransitiveVerb(
    tree: DependencyTree,
    arguments: Seq[(DependencyTree, String)]
  ): Set[Predicate] = {
    val lemma = tree.token.lemma
    val logic =
      for (subjTree <- arguments(0)._1.simplifications;
           subj = subjTree._yield;
           objTree <- arguments(1)._1.simplifications;
           obj = objTree._yield) yield Predicate(lemma, Seq(subj, obj))
    logic.toSet
  }

  def getVerbArguments(tree: DependencyTree) = {
    tree.children.filter(c => {
      val label = c._2
      label == "nsubj" || label == "dobj" || label.startsWith("prep")
    })
  }

}

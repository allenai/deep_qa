package org.allenai.semparse.parse

case class Predicate(predicate: String, arguments: Seq[String])

object LogicalFormGenerator {
  def getLogicalForm(tree: DependencyTree): Set[Predicate] = {
    _getLogicForNode(tree) ++ tree.children.map(_._1).flatMap(_getLogicForNode)
  }

  def _getLogicForNode(tree: DependencyTree): Set[Predicate] = {
    if (tree.isVerb) {
      getLogicForVerb(tree)
    } else {
      Set()
    }
  }

  def getLogicForVerb(tree: DependencyTree): Set[Predicate] = {
    val verbLemma = tree.token.lemma
    val arguments = getVerbArguments(tree)
    if (arguments.size == 1) {
      if (arguments(0)._2 != "nsubj") {
        throw new RuntimeException(s"One argument verb with no subject: ${arguments(0)}")
      }
      val subject = arguments(0)._1
      subject.simplifications.map(s => Predicate(verbLemma, Seq(s._yield)))
    } else if (arguments.size == 2) {
      if (arguments(0)._2 == "nsubj" && arguments(1)._2 == "dobj") {
        val logic =
          for (subjTree <- arguments(0)._1.simplifications;
               subj = subjTree._yield;
               objTree <- arguments(1)._1.simplifications;
               obj = objTree._yield) yield Predicate(verbLemma, Seq(subj, obj))
        logic.toSet
      } else {
        tree.print()
        throw new RuntimeException("unhandled 2-argument verb case")
      }
    } else {
      throw new RuntimeException("verbs with more than two arguments not yet handled")
    }
  }

  def getVerbArguments(tree: DependencyTree) = {
    tree.children.filter(c => {
      val label = c._2
      label == "nsubj" || label == "dobj" || label.startsWith("prep")
    })
  }

}

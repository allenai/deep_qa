package org.allenai.semparse.parse

import scala.collection.mutable

case class Token(word: String, posTag: String, lemma: String, index: Int) {
  override def toString() = s"$word ($lemma): $posTag"
}
case class Dependency(head: String, headIndex: Int, dependent: String, depIndex: Int, label: String)
case class DependencyTree(token: Token, children: Seq[(DependencyTree, String)]) {
  def isNp(): Boolean = token.posTag.startsWith("NN")
  def isVerb(): Boolean = token.posTag.startsWith("VB")

  lazy val childLabels = children.map(_._2).toSet

  lazy val _yield: String = tokens.sortBy(_.index).map(_.word).mkString(" ")

  lazy val tokens: Seq[Token] = Seq(token) ++ children.flatMap(_._1.tokens)

  lazy val simplifications: Set[DependencyTree] = {
    var toRemove = children.sortBy(simplificationSortingKey)
    val simplified = mutable.ArrayBuffer[DependencyTree](this)
    var currentTree = this
    Set(this) ++ toRemove.map(child => {
      currentTree = transformers.removeTree(currentTree, child._1)
      currentTree
    })
  }

  // Anything not shown here will get removed first, then the things at the beginning of this list.
  val simplificationOrder = Seq("prep", "amod")
  def simplificationSortingKey(child: (DependencyTree, String)) = {
    val label = if (child._2.startsWith("prep")) "prep" else child._2
    val labelIndex = simplificationOrder.indexOf(label)
    val tokenIndex = child._1.token.index
    // noun modifiers that come after the noun are more likely to be things to remove first.
    (labelIndex, -tokenIndex)
  }

  def print() {
    _print(1, "ROOT")
  }

  private def _print(level: Int, depLabel: String) {
    for (i <- 1 until level) {
      System.out.print("   ")
    }
    println(s"($depLabel) $token")
    for ((child, label) <- children) {
      child._print(level + 1, label)
    }
  }
}


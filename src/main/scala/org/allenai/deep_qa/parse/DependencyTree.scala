package org.allenai.deep_qa.parse

import scala.collection.mutable

case class Token(word: String, posTag: String, lemma: String, index: Int) {
  override def toString() = s"$word ($lemma): $posTag, $index"

  def addPreposition(prep: String): Token = {
    val newWord = word + "_" + prep
    val newLemma = lemma + "_" + prep
    Token(newWord, posTag, newLemma, index)
  }

  def combineWith(other: Token): Token = {
    val newWord = word + "_" + other.word
    val newLemma = lemma + "_" + other.lemma
    Token(newWord, posTag, newLemma, index)
  }

  def shiftIndexBy(amount: Int): Token = Token(word, posTag, lemma, index + amount)
  def withNewIndex(newIndex: Int): Token = Token(word, posTag, lemma, newIndex)
}
case class Dependency(head: String, headIndex: Int, dependent: String, depIndex: Int, label: String)
case class DependencyTree(token: Token, children: Seq[(DependencyTree, String)]) {
  def isNp(): Boolean = token.posTag.startsWith("NN")
  def isVerb(): Boolean = token.posTag.startsWith("VB")
  def isAdj(): Boolean = token.posTag.startsWith("JJ")
  def isDeterminer(): Boolean = token.posTag.contains("DT")
  def isWhPhrase(): Boolean = token.posTag == "WDT" || token.posTag == "WP" || token.posTag == "WRB"
  def containsWhPhrase(): Boolean = isWhPhrase || children.exists(_._1.containsWhPhrase)
  def shiftIndicesBy(amount: Int): DependencyTree = {
    DependencyTree(token.shiftIndexBy(amount), children.map(c => (c._1.shiftIndicesBy(amount), c._2)))
  }
  def numTokens() = tokensInYield.length
  def tokenStartIndex() = tokensInYield.map(_.index).min

  lazy val childLabels = children.map(_._2).toSet

  // This adds back in tokens for prepositions and possessives, which were stripped when using
  // collapsed dependencies.  And there's a bunch of fancy footwork for dealing with conjunctions,
  // which are rather complicated to fix given the representation we have...
  lazy val tokensInYield: Seq[Token] = {
    val normalizedTree = transformers.RemoveConjunctionDuplicates.transform(this)
    val numberOfConjunctions = normalizedTree.getDescendentsWithLabel("conj_and").size
    var conjunctionsSeen = 0
    val trees = (normalizedTree.children ++ Seq((normalizedTree, "self"))).sortBy(_._1.token.index)
    // NOTE: I'm mutating state in this flatMap!
    trees.flatMap(tree => {
      tree match {
        case t if t._1 == normalizedTree => Seq(token)
        case (child, label) if label.startsWith("prep_") => {
          val prep = label.replace("prep_", "")
          val prepToken = Token(prep, "IN", prep, child.token.index - 1)
          Seq(prepToken) ++ child.tokensInYield
        }
        case (child, "agent") => {
          val prepToken = Token("by", "IN", "by", child.token.index - 1)
          Seq(prepToken) ++ child.tokensInYield
        }
        case (child, "poss") if child.token.posTag != "PRP$" => {
          val possToken = Token("'s", "POS", "'s", child.token.index + 1)
          child.tokensInYield ++ Seq(possToken)
        }
        case (child, "conj_and") => {
          val conjToken = if (conjunctionsSeen == numberOfConjunctions - 1) {
            Token("and", "CC", "and", child.token.index - 1)
          } else {
            Token(",", ",", ",", child.token.index - 1)
          }
          conjunctionsSeen += 1
          Seq(conjToken) ++ child.tokensInYield
        }
        case (child, label) => {
          child.tokensInYield
        }
      }
    })
  }

  // The initial underscore is because "yield" is a reserved word in scala.
  lazy val _yield: String = removeExtraSpacesInYield(tokensInYield.map(_.word).mkString(" "))

  lazy val lemmaYield: String = removeExtraSpacesInYield(tokensInYield.map(_.lemma).mkString(" "))

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

  def getChildWithLabel(label: String): Option[DependencyTree] = {
    val childrenWithLabel = children.filter(_._2 == label)
    if (childrenWithLabel.size == 1) {
      Some(childrenWithLabel.head._1)
    } else {
      None
    }
  }

  def getDescendentsWithLabel(label: String): Seq[DependencyTree] = {
    children.filter(_._2 == label).map(_._1) ++ children.flatMap(_._1.getDescendentsWithLabel(label))
  }

  // Anything not shown here will get removed first, then the things at the beginning of this list.
  val simplificationOrder = Seq("prep", "nn", "amod")
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

  def removeExtraSpacesInYield(yieldStr: String): String = {
    yieldStr.replace(" 's ", "'s ").replace(" , ", ", ")
  }
}


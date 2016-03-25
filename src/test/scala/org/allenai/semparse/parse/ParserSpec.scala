package org.allenai.semparse.parse

import org.scalatest._

class StanfordParserSpec extends FlatSpecLike with Matchers {

  val sentenceParses: Map[String, (Set[Dependency], Seq[Token])] = Map(
    "People eat good food." -> (
      Set(
        Dependency("eat", 2, "People", 1, "nsubj"),
        Dependency("eat", 2, "food", 4, "dobj"),
        Dependency("food", 4, "good", 3, "amod"),
        Dependency("ROOT", 0, "eat", 2, "root")
      ),
      Seq(
        Token("People", "NNS", "people"),
        Token("eat", "VBP", "eat"),
        Token("good", "JJ", "good"),
        Token("food", "NN", "food"),
        Token(".", ".", ".")
      )
    ),
    "Mary went to the store." -> (
      Set(
        Dependency("went", 2, "Mary", 1, "nsubj"),
        Dependency("went", 2, "store", 5, "prep_to"),
        Dependency("store", 5, "the", 4, "det"),
        Dependency("ROOT", 0, "went", 2, "root")
      ),
      Seq(
        Token("Mary", "NNP", "Mary"),
        Token("went", "VBD", "go"),
        Token("to", "TO", "to"),
        Token("the", "DT", "the"),
        Token("store", "NN", "store"),
        Token(".", ".", ".")
      )
    )
  )

  val parser = new StanfordParser

  "parseSentence" should "give correct dependencies and part of speech tags" in {
    val sentence = "People eat good food."
    val parse = parser.parseSentence(sentence)
    val expectedDependencies = sentenceParses(sentence)._1
    val expectedTokens = sentenceParses(sentence)._2

    parse.dependencies.size should be(expectedDependencies.size)
    parse.dependencies.toSet should be(expectedDependencies)

    parse.tokens.size should be(expectedTokens.size)
    parse.tokens should be(expectedTokens)
  }

  it should "give collapsed dependencies" in {
    val sentence = "Mary went to the store."
    val parse = parser.parseSentence(sentence)
    val expectedDependencies = sentenceParses(sentence)._1
    val expectedTokens = sentenceParses(sentence)._2

    parse.dependencies.size should be(expectedDependencies.size)
    parse.dependencies.toSet should be(expectedDependencies)

    parse.tokens.size should be(expectedTokens.size)
    parse.tokens should be(expectedTokens)
  }

  "dependencyTree" should "correctly convert dependencies to graphs" in {
    val sentence = "People eat good food."
    val parse = new ParsedSentence {
      val dependencies = sentenceParses(sentence)._1.toSeq
      val tokens = sentenceParses(sentence)._2
    }
    parse.dependencyTree should be(
      DependencyTree(Token("eat", "VBP", "eat"), Seq(
        (DependencyTree(Token("People", "NNS", "people"), Seq()), "nsubj"),
        (DependencyTree(Token("food", "NN", "food"), Seq(
          (DependencyTree(Token("good", "JJ", "good"), Seq()), "amod"))), "dobj"))))
  }
}

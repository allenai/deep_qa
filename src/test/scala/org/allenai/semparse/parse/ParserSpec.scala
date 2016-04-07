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
        Token("People", "NNS", "people", 1),
        Token("eat", "VBP", "eat", 2),
        Token("good", "JJ", "good", 3),
        Token("food", "NN", "food", 4),
        Token(".", ".", ".", 5)
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
        Token("Mary", "NNP", "mary", 1),
        Token("went", "VBD", "go", 2),
        Token("to", "TO", "to", 3),
        Token("the", "DT", "the", 4),
        Token("store", "NN", "store", 5),
        Token(".", ".", ".", 6)
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

  it should "give collapsed dependencies and lowercase lemmas" in {
    val sentence = "Mary went to the store."
    val parse = parser.parseSentence(sentence)
    val expectedDependencies = sentenceParses(sentence)._1
    val expectedTokens = sentenceParses(sentence)._2

    parse.dependencies.size should be(expectedDependencies.size)
    parse.dependencies.toSet should be(expectedDependencies)

    parse.tokens.size should be(expectedTokens.size)
    parse.tokens should be(expectedTokens)
  }

  "splitSentences" should "return substrings of the original input" in {
    val doc = "This is a test document. It has “funny characters” [and brackets]. Or does it?"
    parser.splitSentences(doc) should be(Seq(
      "This is a test document.",
      "It has “funny characters” [and brackets].",
      "Or does it?"
    ))
  }

  "dependencyTree" should "correctly convert dependencies to graphs" in {
    val sentence = "People eat good food."
    val parse = new ParsedSentence {
      val dependencies = sentenceParses(sentence)._1.toSeq
      val tokens = sentenceParses(sentence)._2
    }
    parse.dependencyTree.get should be(
      DependencyTree(Token("eat", "VBP", "eat", 2), Seq(
        (DependencyTree(Token("People", "NNS", "people", 1), Seq()), "nsubj"),
        (DependencyTree(Token("food", "NN", "food", 4), Seq(
          (DependencyTree(Token("good", "JJ", "good", 3), Seq()), "amod"))), "dobj"))))
  }

  it should "return None if the root has no children" in {
    val parse = new ParsedSentence {
      val dependencies = Seq()
      val tokens = Seq(Token("Eat", "VBD", "eat", 1))
    }
    parse.dependencyTree should be(None)
  }

  it should "return None if there are cycles" in {
    val parse = new ParsedSentence {
      val dependencies = Seq(
        Dependency("ROOT", 0, "Eat", 1, "root"),
        Dependency("Food", 2, "Eat", 1, "crazy-dep"),
        Dependency("Eat", 1, "Food", 2, "crazy-dep")
      )
      val tokens = Seq(Token("Eat", "VBD", "eat", 1), Token("Food", "NN", "food", 2))
    }
    parse.dependencyTree should be(None)
  }
}

package org.allenai.semparse.parse

import org.scalatest._

class StanfordParserSpec extends FlatSpecLike with Matchers {

  val parser = new StanfordParser

  "parseSentence" should "give correct dependencies and part of speech tags" in {
    val sentence = "People eat good food."
    val parse = parser.parseSentence(sentence)

    parse.dependencies.size should be(4)
    parse.dependencies.toSet should be(Set(
      Dependency("eat", 2, "People", 1, "nsubj"),
      Dependency("eat", 2, "food", 4, "dobj"),
      Dependency("food", 4, "good", 3, "amod"),
      Dependency("ROOT", 0, "eat", 2, "root")))

    parse.posTags.size should be(5)
    parse.posTags(0) should be(PartOfSpeech("People", "NNS"))
    parse.posTags(1) should be(PartOfSpeech("eat", "VBP"))
    parse.posTags(2) should be(PartOfSpeech("good", "JJ"))
    parse.posTags(3) should be(PartOfSpeech("food", "NN"))
    parse.posTags(4) should be(PartOfSpeech(".", "."))
  }

  it should "give collapsed dependencies" in {
    val sentence = "Mary went to the store."
    val parse = parser.parseSentence(sentence)

    parse.dependencies.size should be(4)
    parse.dependencies.toSet should be(Set(
      Dependency("went", 2, "Mary", 1, "nsubj"),
      Dependency("went", 2, "store", 5, "prep_to"),
      Dependency("store", 5, "the", 4, "det"),
      Dependency("ROOT", 0, "went", 2, "root")))

  }

  "dependencyTree" should "correctly convert dependencies to graphs" in {
    val parse = new ParsedSentence {
      val dependencies = Seq(
        Dependency("eat", 2, "People", 1, "nsubj"),
        Dependency("eat", 2, "food", 4, "dobj"),
        Dependency("food", 4, "good", 3, "amod"),
        Dependency("ROOT", 0, "eat", 2, "root")
      )
      val posTags = Seq(
        PartOfSpeech("People", "NNS"),
        PartOfSpeech("eat", "VBP"),
        PartOfSpeech("good", "JJ"),
        PartOfSpeech("food", "NN"),
        PartOfSpeech(".", ".")
      )
    }
    parse.dependencyTree should be(
      DependencyTree("eat", "VBP", Seq(
        (DependencyTree("People", "NNS", Seq()), "nsubj"),
        (DependencyTree("food", "NN", Seq(
          (DependencyTree("good", "JJ", Seq()), "amod"))), "dobj"))))
  }

  "just playing" should "give parses for these sentences" in {
    val sentences = Seq(
      "Cells contain genetic material called DNA.",
      "Most of Earth is covered by water.",
      "Humans depend on plants for oxygen.",
      "The seeds of an oak come from the fruit.",
      "Which gas is given off by plants?",
      "Matter that is vibrating is producing sound.",
      "Which of these is an example of liquid water?"
    )
    for (sentence <- sentences) {
      println("")
      parser.parseSentence(sentence).dependencyTree.print
      println("")
    }
  }
}

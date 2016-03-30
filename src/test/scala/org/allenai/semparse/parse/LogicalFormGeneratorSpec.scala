package org.allenai.semparse.parse

import org.scalatest._

class LogicalFormGeneratorSpec extends FlatSpecLike with Matchers {
  val parser = new StanfordParser
  "getLogicalForm" should "work for \"Cells contain genetic material called DNA.\"" in {
    val parse = parser.parseSentence("Cells contain genetic material called DNA.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should contain theSameElementsAs Set(
      Predicate("contain", Seq("cell", "genetic material call dna")),
      Predicate("contain", Seq("cell", "genetic material")),
      Predicate("contain", Seq("cell", "material")),
      Predicate("genetic", Seq("material")),
      Predicate("call", Seq("genetic material", "dna")),
      Predicate("call", Seq("material", "dna"))
    )
  }

  it should "work for \"Most of Earth is covered by water.\"" in {
    val parse = parser.parseSentence("Most of Earth is covered by water.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should contain theSameElementsAs Set(
      Predicate("cover", Seq("water", "most of Earth")),
      Predicate("cover", Seq("water", "most")),
      Predicate("of", Seq("most", "Earth"))
    )
  }

  it should "work for \"Humans depend on plants for oxygen.\"" in {
    // Unfortunately, the Stanford parser gives an incorrect dependency parse for this sentence, so
    // the logical form is not as we would really want...
    val parse = parser.parseSentence("Humans depend on plants for oxygen.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should contain theSameElementsAs Set(
      Predicate("depend_on", Seq("human", "plant for oxygen")),
      Predicate("depend_on", Seq("human", "plant")),
      Predicate("for", Seq("plant", "oxygen"))
    )
  }

  it should "work for \"Humans depend on plants for oxygen.\" (with a correct parse)" in {
    val tree =
      DependencyTree(Token("depend", "VBP", "depend", 2), Seq(
        (DependencyTree(Token("Humans", "NNS", "human", 1), Seq()), "nsubj"),
        (DependencyTree(Token("plants", "NNS", "plant", 4), Seq()), "prep_on"),
        (DependencyTree(Token("oxygen", "NNS", "oxygen", 6), Seq()), "prep_for")))
    LogicalFormGenerator.getLogicalForm(tree) should contain theSameElementsAs Set(
      Predicate("depend_on", Seq("human", "plant")),
      Predicate("depend_for", Seq("human", "oxygen")),
      Predicate("depend_for_on", Seq("oxygen", "plant"))  // preps are sorted alphabetically
    )
  }

  it should "work for \"The seeds of an oak come from the fruit.\"" in {
    val parse = parser.parseSentence("The seeds of an oak come from the fruit.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should contain theSameElementsAs Set(
      Predicate("come_from", Seq("seed of oak", "fruit")),
      Predicate("come_from", Seq("seed", "fruit")),
      Predicate("of", Seq("seed", "oak"))
    )
  }

  it should "work for \"Which gas is given off by plants?\"" in {
    val parse = parser.parseSentence("Which gas is given off by plants?")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should contain theSameElementsAs Set(
      Predicate("give_off", Seq("plant", "gas"))
    )
  }

  it should "work for \"MOST erosion at a beach is caused by waves.\"" in {
    val parse = parser.parseSentence("MOST erosion at a beach is caused by waves.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should contain theSameElementsAs Set(
      Predicate("cause", Seq("wave", "erosion at beach")),
      Predicate("cause", Seq("wave", "erosion")),
      Predicate("at", Seq("erosion", "beach"))
    )
  }

  it should "work for \"Water causes the most soil and rock erosion.\"" in {
    // Once again, the Stanford parser is wrong here...
    val parse = parser.parseSentence("Water causes the most soil and rock erosion.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should contain theSameElementsAs Set(
      Predicate("cause", Seq("water", "soil")),
      Predicate("cause", Seq("water", "rock erosion")),
      Predicate("cause", Seq("water", "erosion")),
      Predicate("rock", Seq("erosion"))
    )
  }

  it should "work for \"Water causes the most soil and rock erosion.\" (with a correct parse)" in {
    val tree =
      DependencyTree(Token("causes", "VBZ", "cause", 2), Seq(
        (DependencyTree(Token("Water", "NN", "water", 1), Seq()), "nsubj"),
        (DependencyTree(Token("erosion", "NN", "erosion", 8), Seq(
          (DependencyTree(Token("the", "DT", "the", 3), Seq()), "det"),
          (DependencyTree(Token("most", "JJS", "most", 4), Seq()), "amod"),
          (DependencyTree(Token("soil", "NN", "soil", 5), Seq(
            (DependencyTree(Token("rock", "NN", "rock", 7), Seq()), "conj_and"))), "nn"),
          (DependencyTree(Token("rock", "NN", "rock", 7), Seq()), "nn"))), "dobj")))
    LogicalFormGenerator.getLogicalForm(tree) should contain theSameElementsAs Set(
      Predicate("cause", Seq("water", "soil erosion")),
      Predicate("cause", Seq("water", "rock erosion")),
      Predicate("cause", Seq("water", "erosion")),
      Predicate("soil", Seq("erosion")),
      Predicate("rock", Seq("erosion"))
    )
  }

  it should "work for \"Roots can slow down erosion.\"" in {
    val parse = parser.parseSentence("Roots can slow down erosion.")
    parse.dependencyTree.print()
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should contain theSameElementsAs Set(
      Predicate("slow_down", Seq("root", "erosion"))
    )
  }
}

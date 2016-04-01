package org.allenai.semparse.parse

import org.scalatest._

class LogicalFormGeneratorSpec extends FlatSpecLike with Matchers {
  val parser = new StanfordParser
  "getLogicalForm" should "work for \"Cells contain genetic material called DNA.\"" in {
    val parse = parser.parseSentence("Cells contain genetic material called DNA.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
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
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("cover", Seq("water", "most of earth")),
      Predicate("cover", Seq("water", "most")),
      Predicate("of", Seq("most", "earth"))
    )
  }

  it should "work for \"Humans depend on plants for oxygen.\"" in {
    // Unfortunately, the Stanford parser gives an incorrect dependency parse for this sentence, so
    // the logical form is not as we would really want...
    val parse = parser.parseSentence("Humans depend on plants for oxygen.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
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
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("come_from", Seq("seed of oak", "fruit")),
      Predicate("come_from", Seq("seed", "fruit")),
      Predicate("of", Seq("seed", "oak"))
    )
  }

  it should "work for \"Which gas is given off by plants?\"" in {
    val parse = parser.parseSentence("Which gas is given off by plants?")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("give_off", Seq("plant", "gas"))
    )
  }

  it should "work for \"MOST erosion at a beach is caused by waves.\"" in {
    val parse = parser.parseSentence("MOST erosion at a beach is caused by waves.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("cause", Seq("wave", "erosion at beach")),
      Predicate("cause", Seq("wave", "erosion")),
      Predicate("at", Seq("erosion", "beach"))
    )
  }

  it should "work for \"Water causes the most soil and rock erosion.\"" in {
    // Once again, the Stanford parser is wrong here...
    val parse = parser.parseSentence("Water causes the most soil and rock erosion.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
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
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("slow_down", Seq("root", "erosion"))
    )
  }

  it should "work for \"regulatory mechanisms of photosynthesis.\"" in {
    val parse = parser.parseSentence("regulatory mechanisms of photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("regulatory", Seq("mechanism")),
      Predicate("of", Seq("regulatory mechanism", "photosynthesis")),
      Predicate("of", Seq("mechanism", "photosynthesis"))
    )
  }

  it should "work for \"an example is photosynthesis.\"" in {
    val parse = parser.parseSentence("an example is photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("be", Seq("photosynthesis", "example"))
    )
  }

  it should "work for \"Manganese is necessary for photosynthesis.\"" in {
    val parse = parser.parseSentence("Manganese is necessary for photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("necessary_for", Seq("manganese", "photosynthesis"))
    )
  }

  it should "work for \"Oxygen and energy are products of photosynthesis.\"" in {
    val parse = parser.parseSentence("Oxygen and energy are products of photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("product_of", Seq("oxygen", "photosynthesis")),
      Predicate("product_of", Seq("energy", "photosynthesis"))
    )
  }

  it should "work for \"But, photosynthesis is older.\"" in {
    val parse = parser.parseSentence("But, photosynthesis is older.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("be", Seq("older", "photosynthesis"))
    )
  }

  it should "work for \"This process, photosynthesis, liberates oxygen.\"" in {
    val parse = parser.parseSentence("This process, photosynthesis, liberates oxygen.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("liberate", Seq("photosynthesis", "oxygen")),
      Predicate("liberate", Seq("process", "oxygen"))
    )
  }

  it should "work for \"Plants generate oxygen during photosynthesis.\"" in {
    val parse = parser.parseSentence("Plants generate oxygen during photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("generate", Seq("plant", "oxygen")),
      Predicate("generate_during", Seq("plant", "photosynthesis")),
      Predicate("generate_obj_during", Seq("oxygen", "photosynthesis"))
    )
  }

  it should "work for \"An example would be photosynthesis.\"" in {
    val parse = parser.parseSentence("An example would be photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("be", Seq("photosynthesis", "example"))
    )
  }

  it should "work for \"Photosynthesis is impossible without light.\"" in {
    val parse = parser.parseSentence("Photosynthesis is impossible without light.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("impossible_without", Seq("photosynthesis", "light"))
    )
  }

  it should "work for \"Food originates with photosynthesis .\"" in {
    val parse = parser.parseSentence("Food originates with photosynthesis .")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("originate_with", Seq("food", "photosynthesis"))
    )
  }

  it should "work for \"There are two stages of photosynthesis.\"" in {
    val parse = parser.parseSentence("There are two stages of photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("exists", Seq("two stage of photosynthesis")),
      Predicate("exists", Seq("stage of photosynthesis")),
      Predicate("exists", Seq("stage")),
      Predicate("of", Seq("two stage", "photosynthesis")),
      Predicate("of", Seq("stage", "photosynthesis"))
    )
  }

  it should "work for \"Photosynthesis gives a plant energy.\"" in {
    val parse = parser.parseSentence("Photosynthesis gives a plant energy.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("give", Seq("photosynthesis", "energy")),
      Predicate("give_obj2", Seq("photosynthesis", "plant")),
      Predicate("give_obj_obj2", Seq("energy", "plant"))
    )
  }

  it should "work for \"Diuron works by inhibiting photosynthesis.\"" in {
    val parse = parser.parseSentence("Diuron works by inhibiting photosynthesis.")
    parse.dependencyTree.get.print()
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("work_C_inhibit", Seq("Diuron", "photosynthesis"))
    )
  }

  it should "work for \"Diuron works to inhibit photosynthesis.\"" in {
    val parse = parser.parseSentence("Diuron works to inhibit photosynthesis.")
    parse.dependencyTree.get.print()
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("work_C_inhibit", Seq("Diuron", "photosynthesis"))
    )
  }

  it should "work for \"Sue asked Bill to stop.\"" in {
    val parse = parser.parseSentence("Sue asked Bill to stop.")
    parse.dependencyTree.get.print()
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should contain theSameElementsAs Set(
      Predicate("ask_C_stop", Seq("Sue", "Bill"))
    )
  }
}

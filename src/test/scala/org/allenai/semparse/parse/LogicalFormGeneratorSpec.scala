package org.allenai.semparse.parse

import org.scalatest._

class LogicalFormGeneratorSpec extends FlatSpecLike with Matchers {
  val parser = new StanfordParser
  "getLogicalForm" should "work for \"Cells contain genetic material called DNA.\" (simplifications)" in {
    val parse = parser.parseSentence("Cells contain genetic material called DNA.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("contain", Seq(Atom("cell"), Atom("genetic material call dna"))),
      Predicate("contain", Seq(Atom("cell"), Atom("genetic material"))),
      Predicate("contain", Seq(Atom("cell"), Atom("material"))),
      Predicate("genetic", Seq(Atom("material"))),
      Predicate("call", Seq(Atom("genetic material"), Atom("dna"))),
      Predicate("call", Seq(Atom("material"), Atom("dna")))
    ))))
  }

  it should "work for \"Most of Earth is covered by water.\" (passives, simplifications)" in {
    val parse = parser.parseSentence("Most of Earth is covered by water.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("cover", Seq(Atom("water"), Atom("most of earth"))),
      Predicate("cover", Seq(Atom("water"), Atom("most"))),
      Predicate("of", Seq(Atom("most"), Atom("earth")))
    ))))
  }

  it should "work for \"Humans depend on plants for oxygen.\" (verbs with prepositions)" in {
    // Unfortunately, the Stanford parser gives an incorrect dependency parse for this sentence, so
    // the logical form is not as we would really want...
    val parse = parser.parseSentence("Humans depend on plants for oxygen.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("depend_on", Seq(Atom("human"), Atom("plant for oxygen"))),
      Predicate("depend_on", Seq(Atom("human"), Atom("plant"))),
      Predicate("for", Seq(Atom("plant"), Atom("oxygen")))
    ))))
  }

  it should "work for \"Humans depend on plants for oxygen.\" (with a correct parse; verbs with two prepositions)" in {
    val tree =
      DependencyTree(Token("depend", "VBP", "depend", 2), Seq(
        (DependencyTree(Token("Humans", "NNS", "human", 1), Seq()), "nsubj"),
        (DependencyTree(Token("plants", "NNS", "plant", 4), Seq()), "prep_on"),
        (DependencyTree(Token("oxygen", "NNS", "oxygen", 6), Seq()), "prep_for")))
    LogicalFormGenerator.getLogicalForm(tree) should be(Some(Conjunction(Set(
      Predicate("depend_on", Seq(Atom("human"), Atom("plant"))),
      Predicate("depend_for", Seq(Atom("human"), Atom("oxygen"))),
      Predicate("depend_for_on", Seq(Atom("oxygen"), Atom("plant")))  // preps are sorted alphabetically
    ))))
  }

  it should "work for \"The seeds of an oak come from the fruit.\" (dropping determiners)" in {
    val parse = parser.parseSentence("The seeds of an oak come from the fruit.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("come_from", Seq(Atom("seed of oak"), Atom("fruit"))),
      Predicate("come_from", Seq(Atom("seed"), Atom("fruit"))),
      Predicate("of", Seq(Atom("seed"), Atom("oak")))
    ))))
  }

  it should "work for \"Which gas is given off by plants?\" (more dropping determiners)" in {
    val parse = parser.parseSentence("Which gas is given off by plants?")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("give_off", Seq(Atom("plant"), Atom("gas")))
    ))))
  }

  it should "work for \"MOST erosion at a beach is caused by waves.\" (dropping superlatives)" in {
    val parse = parser.parseSentence("MOST erosion at a beach is caused by waves.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("cause", Seq(Atom("wave"), Atom("erosion at beach"))),
      Predicate("cause", Seq(Atom("wave"), Atom("erosion"))),
      Predicate("at", Seq(Atom("erosion"), Atom("beach")))
    ))))
  }

  it should "work for \"Water causes the most soil and rock erosion.\" (conjunctions)" in {
    // Once again, the Stanford parser is wrong here...
    val parse = parser.parseSentence("Water causes the most soil and rock erosion.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("cause", Seq(Atom("water"), Atom("soil"))),
      Predicate("cause", Seq(Atom("water"), Atom("rock erosion"))),
      Predicate("cause", Seq(Atom("water"), Atom("erosion"))),
      Predicate("rock", Seq(Atom("erosion")))
    ))))
  }

  it should "work for \"Water causes the most soil and rock erosion.\" (with a correct parse; conjunctions)" in {
    val tree =
      DependencyTree(Token("causes", "VBZ", "cause", 2), Seq(
        (DependencyTree(Token("Water", "NN", "water", 1), Seq()), "nsubj"),
        (DependencyTree(Token("erosion", "NN", "erosion", 8), Seq(
          (DependencyTree(Token("the", "DT", "the", 3), Seq()), "det"),
          (DependencyTree(Token("most", "JJS", "most", 4), Seq()), "amod"),
          (DependencyTree(Token("soil", "NN", "soil", 5), Seq(
            (DependencyTree(Token("rock", "NN", "rock", 7), Seq()), "conj_and"))), "nn"),
          (DependencyTree(Token("rock", "NN", "rock", 7), Seq()), "nn"))), "dobj")))
    LogicalFormGenerator.getLogicalForm(tree) should be(Some(Conjunction(Set(
      Predicate("cause", Seq(Atom("water"), Atom("soil erosion"))),
      Predicate("cause", Seq(Atom("water"), Atom("rock erosion"))),
      Predicate("cause", Seq(Atom("water"), Atom("erosion"))),
      Predicate("soil", Seq(Atom("erosion"))),
      Predicate("rock", Seq(Atom("erosion")))
    ))))
  }

  it should "work for \"Roots can slow down erosion.\" (dropping modals)" in {
    val parse = parser.parseSentence("Roots can slow down erosion.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("slow_down", Seq(Atom("root"), Atom("erosion")))
    ))))
  }

  it should "work for \"regulatory mechanisms of photosynthesis.\"" in {
    val parse = parser.parseSentence("regulatory mechanisms of photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("regulatory", Seq(Atom("mechanism"))),
      Predicate("of", Seq(Atom("regulatory mechanism"), Atom("photosynthesis"))),
      Predicate("of", Seq(Atom("mechanism"), Atom("photosynthesis")))
    ))))
  }

  it should "work for \"an example is photosynthesis.\" (copula)" in {
    val parse = parser.parseSentence("an example is photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("be", Seq(Atom("photosynthesis"), Atom("example")))
    ))))
  }

  it should "work for \"Manganese is necessary for photosynthesis.\" (X BE Y PREP Z)" in {
    val parse = parser.parseSentence("Manganese is necessary for photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("necessary_for", Seq(Atom("manganese"), Atom("photosynthesis")))
    ))))
  }

  it should "work for \"Oxygen and energy are products of photosynthesis.\" (conjunctions with X BE Y PREP Z)" in {
    val parse = parser.parseSentence("Oxygen and energy are products of photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("product_of", Seq(Atom("oxygen"), Atom("photosynthesis"))),
      Predicate("product_of", Seq(Atom("energy"), Atom("photosynthesis")))
    ))))
  }

  it should "work for \"But, photosynthesis is older.\" (drop CCs)" in {
    val parse = parser.parseSentence("But, photosynthesis is older.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("be", Seq(Atom("older"), Atom("photosynthesis")))
    ))))
  }

  it should "work for \"This process, photosynthesis, liberates oxygen.\" (appositives)" in {
    val parse = parser.parseSentence("This process, photosynthesis, liberates oxygen.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("liberate", Seq(Atom("photosynthesis"), Atom("oxygen"))),
      Predicate("liberate", Seq(Atom("process"), Atom("oxygen"))),
      Predicate("appos", Seq(Atom("process"), Atom("photosynthesis")))
    ))))
  }

  it should "work for \"Plants generate oxygen during photosynthesis.\" (verb with subj, obj, and prep)" in {
    val parse = parser.parseSentence("Plants generate oxygen during photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("generate", Seq(Atom("plant"), Atom("oxygen"))),
      Predicate("generate_during", Seq(Atom("plant"), Atom("photosynthesis"))),
      Predicate("generate_obj_during", Seq(Atom("oxygen"), Atom("photosynthesis")))
    ))))
  }

  it should "work for \"An example would be photosynthesis.\" (copula with modal)" in {
    val parse = parser.parseSentence("An example would be photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("be", Seq(Atom("photosynthesis"), Atom("example")))
    ))))
  }

  it should "work for \"Photosynthesis is impossible without light.\" (X BE Y PREP Z)" in {
    val parse = parser.parseSentence("Photosynthesis is impossible without light.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("impossible_without", Seq(Atom("photosynthesis"), Atom("light")))
    ))))
  }

  it should "work for \"Food originates with photosynthesis .\" (lowercase NNPs)" in {
    // Stanford says "Food" is an NNP in this sentence.
    val parse = parser.parseSentence("Food originates with photosynthesis .")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("originate_with", Seq(Atom("food"), Atom("photosynthesis")))
    ))))
  }

  it should "work for \"There are two stages of photosynthesis.\" (existentials)" in {
    val parse = parser.parseSentence("There are two stages of photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("exists", Seq(Atom("two stage of photosynthesis"))),
      Predicate("exists", Seq(Atom("stage of photosynthesis"))),
      Predicate("exists", Seq(Atom("stage"))),
      Predicate("of", Seq(Atom("two stage"), Atom("photosynthesis"))),
      Predicate("of", Seq(Atom("stage"), Atom("photosynthesis")))
    ))))
  }

  it should "work for \"Photosynthesis gives a plant energy.\" (verb with subj, obj, and obj2)" in {
    val parse = parser.parseSentence("Photosynthesis gives a plant energy.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("give", Seq(Atom("photosynthesis"), Atom("energy"))),
      Predicate("give_obj2", Seq(Atom("photosynthesis"), Atom("plant"))),
      Predicate("give_obj_obj2", Seq(Atom("energy"), Atom("plant")))
    ))))
  }

  it should "work for \"Diuron works by inhibiting photosynthesis.\" (dependent clause with preposition)" in {
    val parse = parser.parseSentence("Diuron works by inhibiting photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("work_by_inhibit", Seq(Atom("diuron"), Atom("photosynthesis")))
    ))))
  }

  it should "work for \"Diuron works to inhibit photosynthesis.\" (controlled subject)" in {
    val parse = parser.parseSentence("Diuron works to inhibit photosynthesis.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("work_Csubj_inhibit_obj", Seq(Atom("diuron"), Atom("photosynthesis")))
    ))))
  }

  it should "work for \"Sam uses tools to measure things.\" (controlled subject with arguments)" in {
    val parse = parser.parseSentence("Sam uses tools to measure things.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("use_Csubj_measure", Seq(Atom("sam"), Atom("tool"))),
      Predicate("use_Csubj_measure_obj", Seq(Atom("sam"), Atom("thing"))),
      Predicate("use_Csubj_measure_obj_obj", Seq(Atom("tool"), Atom("thing")))
    ))))
  }

  it should "work for \"Tools are used to measure things.\" (passive controlled subject with arguments)" in {
    val parse = parser.parseSentence("Tools are used to measure things.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("use_Csubj_measure_obj_obj", Seq(Atom("tool"), Atom("thing")))
    ))))
  }

  it should "work for \"Sue asked Bill to stop.\" (controlled object)" in {
    val parse = parser.parseSentence("Sue asked Bill to stop.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("ask_Cobj_stop", Seq(Atom("sue"), Atom("bill")))
    ))))
  }

  it should "work for \"Sue asked Bill to stop the car.\" (controlled object with arguments)" in {
    val parse = parser.parseSentence("Sue asked Bill to stop the car.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("ask_Cobj_stop", Seq(Atom("sue"), Atom("bill"))),
      Predicate("ask_Cobj_stop_obj", Seq(Atom("sue"), Atom("car"))),
      Predicate("ask_Cobj_stop_obj_obj", Seq(Atom("bill"), Atom("car")))
    ))))
  }

  it should "work for \"The plant produces clusters which contain sperm.\" (relative clause with which)" in {
    val parse = parser.parseSentence("The plant produces clusters which contain sperm.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("produce", Seq(Atom("plant"), Atom("cluster which contain sperm"))),
      Predicate("produce", Seq(Atom("plant"), Atom("cluster"))),
      Predicate("contain", Seq(Atom("cluster"), Atom("sperm")))
    ))))
  }

  it should "work for \"There are many ants that eat food.\" (relative clause with that)" in {
    val parse = parser.parseSentence("There are many ants that eat food.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("exists", Seq(Atom("many ant that eat food"))),
      Predicate("exists", Seq(Atom("many ant"))),
      Predicate("many", Seq(Atom("ant"))),
      Predicate("exists", Seq(Atom("ant"))),
      Predicate("eat", Seq(Atom("many ant"), Atom("food"))),
      Predicate("eat", Seq(Atom("ant"), Atom("food")))
    ))))
  }

  it should "work for \"Plant growth in turn sustains animal life.\" (simplification order)" in {
    val parse = parser.parseSentence("Plant growth in turn sustains animal life.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("sustain", Seq(Atom("plant growth in turn"), Atom("animal life"))),
      Predicate("sustain", Seq(Atom("plant growth"), Atom("animal life"))),
      Predicate("sustain", Seq(Atom("growth"), Atom("animal life"))),
      Predicate("sustain", Seq(Atom("plant growth in turn"), Atom("life"))),
      Predicate("sustain", Seq(Atom("plant growth"), Atom("life"))),
      Predicate("sustain", Seq(Atom("growth"), Atom("life"))),
      Predicate("in", Seq(Atom("plant growth"), Atom("turn"))),
      Predicate("in", Seq(Atom("growth"), Atom("turn"))),
      Predicate("plant", Seq(Atom("growth"))),
      Predicate("animal", Seq(Atom("life")))
    ))))
  }

  it should "work for \"A spam from LOGIC@imneverwrong.com, a particularly blatant crank.\" (weird error)" in {
    val parse = parser.parseSentence("A spam from LOGIC@imneverwrong.com, a particularly blatant crank.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree.get) should be(Some(Conjunction(Set(
      Predicate("from", Seq(Atom("spam"), Atom("logic@imneverwrong.com, particularly blatant"))),
      Predicate("from", Seq(Atom("spam"), Atom("logic@imneverwrong.com,"))),
      Predicate("blatant", Seq(Atom("logic@imneverwrong.com,")))
    ))))
  }

}

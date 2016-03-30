package org.allenai.semparse.parse

import org.scalatest._

class LogicalFormGeneratorSpec extends FlatSpecLike with Matchers {
  val parser = new StanfordParser
  "getLogicalForm" should "work for \"Cells contain genetic material called DNA.\"" in {
    val parse = parser.parseSentence("Cells contain genetic material called DNA.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should be(Set(
      Predicate("contain", Seq("Cells", "genetic material called DNA")),
      Predicate("contain", Seq("Cells", "genetic material")),
      Predicate("contain", Seq("Cells", "material")),
      Predicate("genetic", Seq("material")),
      Predicate("call", Seq("genetic material", "DNA")),
      Predicate("call", Seq("material", "DNA"))
    ))
  }

  it should "work for \"Most of Earth is covered by water.\"" in {
    val parse = parser.parseSentence("Most of Earth is covered by water.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should be(Set(
      Predicate("cover", Seq("water", "Most of Earth")),
      Predicate("cover", Seq("water", "Most")),
      Predicate("of", Seq("Most", "Earth"))
    ))
  }

  it should "work for \"Humans depend on plants for oxygen.\"" in {
    // Unfortunately, the Stanford parser gives an incorrect dependency parse for this sentence, so
    // the logical form is not as we would really want...
    val parse = parser.parseSentence("Humans depend on plants for oxygen.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should be(Set(
      Predicate("depend_on", Seq("Humans", "plants for oxygen")),
      Predicate("depend_on", Seq("Humans", "plants")),
      Predicate("for", Seq("plants", "oxygen"))
    ))
  }

  it should "work for \"The seeds of an oak come from the fruit.\"" in {
    val parse = parser.parseSentence("The seeds of an oak come from the fruit.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should be(Set(
      Predicate("come_from", Seq("seeds of oak", "fruit")),
      Predicate("come_from", Seq("seeds", "fruit")),
      Predicate("of", Seq("seeds", "oak"))
    ))
  }

  it should "work for \"Which gas is given off by plants?\"" in {
    val parse = parser.parseSentence("Which gas is given off by plants?")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should be(Set(
      Predicate("give_off", Seq("plants", "gas"))
    ))
  }
}

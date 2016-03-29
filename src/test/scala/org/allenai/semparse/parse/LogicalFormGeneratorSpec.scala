package org.allenai.semparse.parse

import org.scalatest._

class LogicalFormGeneratorSpec extends FlatSpecLike with Matchers {
  val parser = new StanfordParser
  "getLogicalForm" should "get correct forms for some simple examples" in {
    var parse = parser.parseSentence("Cells contain genetic material called DNA.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should be(Set(
      Predicate("contain", Seq("Cells", "genetic material called DNA")),
      Predicate("contain", Seq("Cells", "genetic material")),
      Predicate("contain", Seq("Cells", "material")),
      Predicate("genetic", Seq("material")),
      Predicate("called", Seq("genetic material", "DNA")),
      Predicate("called", Seq("material", "DNA"))
    ))

    parse = parser.parseSentence("Most of Earth is covered by water.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should be(Set(
      Predicate("cover", Seq("water", "Most of Earth")),
      Predicate("cover", Seq("water", "Most")),
      Predicate("of", Seq("Most", "Earth"))
    ))

    parse = parser.parseSentence("Humans depend on plants for oxygen.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should be(Set(
      Predicate("depend_on", Seq("Humans", "plants")),
      Predicate("depend_for", Seq("Humans", "oxygen")),
      Predicate("depend_on_for", Seq("plants", "oxygen"))
    ))

    parse = parser.parseSentence("The seeds of an oak come from the fruit.")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should be(Set(
      Predicate("come_from", Seq("seeds of oak", "fruit")),
      Predicate("come_from", Seq("seeds", "fruit")),
      Predicate("of", Seq("seeds", "oak"))
    ))

    parse = parser.parseSentence("Which gas is given off by plants?")
    LogicalFormGenerator.getLogicalForm(parse.dependencyTree) should be(Set(
      Predicate("give_off", Seq("plants", "Which gas")),
      Predicate("give_off", Seq("plants", "gas"))
    ))
  }
}

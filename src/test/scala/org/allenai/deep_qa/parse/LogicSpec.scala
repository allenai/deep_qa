package org.allenai.deep_qa.parse

import org.scalatest._

class LogicSpec extends FlatSpecLike with Matchers {
  "fromString" should "parse single atoms" in {
    val logic = Atom("atom")
    Logic.fromString(logic.toString) should be(logic)
  }

  it should "parse unary predicates" in {
    val logic = Predicate("company", Seq(Atom("pixar")))
    Logic.fromString(logic.toString) should be(logic)
  }

  it should "parse binary predicates" in {
    val logic = Predicate("from", Seq(Atom("Bush"), Atom("Texas")))
    Logic.fromString(logic.toString) should be(logic)
  }

  it should "parse ternary predicates" in {
    val logic = Predicate("depend", Seq(Atom("humans"), Atom("plants"), Atom("energy")))
    Logic.fromString(logic.toString) should be(logic)
  }

  it should "handle \"and\" specially" in {
    val logic = Conjunction(Set(Atom("humans"), Atom("plants"), Atom("energy")))
    Logic.fromString(logic.toString) should be(logic)
  }

  it should "handle nesting with conjunctions" in {
    val logic = Predicate("for", Seq(
      Predicate("depend_on", Seq(Atom("humans"), Conjunction(Set(Atom("plants"), Atom("animals"))))),
      Atom("energy")))
    Logic.fromString(logic.toString) should be(logic)
  }
}

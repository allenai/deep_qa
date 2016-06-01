package org.allenai.semparse.one_off

import org.scalatest._

import org.allenai.semparse.parse.Atom
import org.allenai.semparse.parse.Conjunction
import org.allenai.semparse.parse.Predicate

class debug_format_to_training_data_spec extends FlatSpecLike with Matchers {

  "parseDebugLine" should "correctly process the line" in {
    val line = "Photosynthesis is ... -> spread_in(photosynthesis, nature) test(args)"
    val sentence = "Photosynthesis is ..."
    val logic = Conjunction(Set(Predicate("spread_in", Seq(Atom("photosynthesis"), Atom("nature"))),
      Predicate("test", Seq(Atom("args")))))
    debug_format_to_training_data.parseDebugLine(line) should be((sentence, Some(logic)))
  }

  it should "work with empty lfs" in {
    val line = "Photosynthesis is ... -> "
    val sentence = "Photosynthesis is ..."
    val logic = None
    debug_format_to_training_data.parseDebugLine(line) should be((sentence, logic))
  }
}

package org.allenai.semparse.pipeline.science_data

import org.scalatest._

import org.allenai.semparse.parse.Atom
import org.allenai.semparse.parse.Predicate

class ScienceSentenceProcessorSpec extends FlatSpecLike with Matchers {
  "logicalFormToTrainingData" should "produce correct output when the entities have commas" in {
    val predicates = Set(
      Predicate("from", Seq(Atom("spam"), Atom("logic@imneverwrong.com, particularly blatant"))),
      Predicate("from", Seq(Atom("spam"), Atom("logic@imneverwrong.com,"))),
      Predicate("blatant", Seq(Atom("logic@imneverwrong.com,")))
    )
    val sentence = "sentence"

    val json = "[[\"from\",\"spam\",\"logic@imneverwrong.com, particularly blatant\"], [\"from\",\"spam\",\"logic@imneverwrong.com,\"], [\"blatant\",\"logic@imneverwrong.com,\"]]"
    val expectedStrings = Seq(
      "\t\"spam\" \"logic@imneverwrong.com, particularly blatant\"\t\tfrom\t((word-rel \"from\") \"spam\" \"logic@imneverwrong.com, particularly blatant\")\t" + json + "\t" + sentence,
      "\t\"spam\" \"logic@imneverwrong.com,\"\t\tfrom\t((word-rel \"from\") \"spam\" \"logic@imneverwrong.com,\")\t" + json + "\t" + sentence,
      "\t\"logic@imneverwrong.com,\"\tblatant\t\t((word-cat \"blatant\") \"logic@imneverwrong.com,\")\t" + json + "\t" + sentence
    )

    val strings = Helper.logicalFormToTrainingData((sentence, predicates))
    strings.size should be(expectedStrings.size)
    for (i <- 0 until expectedStrings.size) {
      strings(i) should be(expectedStrings(i))
    }
  }
}

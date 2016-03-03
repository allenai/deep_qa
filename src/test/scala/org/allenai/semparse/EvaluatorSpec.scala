package org.allenai.semparse

import org.scalatest._

import edu.cmu.ml.rtw.users.matt.util.FakeFileUtil

class EvaluationSpec extends FlatSpecLike with Matchers {

  val dataFile = "/data/file"
  val dataFileContents = """/m/01cb_k /m/03z3fl	"H. pylori" "J. Robin Warren"		discovered_by	((word-rel "discovered_by") "/m/01cb_k" "/m/03z3fl")	[["discovered_by","/m/01cb_k","/m/03z3fl"], ["discovered_by","/m/01cb_k","/m/023fbk"]]
/m/0gkq30	"Car Service"	best		((word-cat "best") "/m/0gkq30")	[["best","/m/0gkq30"]]
/m/03rt9	"Ireland"	the		((word-cat "the") "/m/03rt9")	[["throughout","/m/03rt9"], ["the","/m/03rt9"], ["uk,","/m/03rt9"], ["operating","/m/03rt9"]]
/m/0f8l9c /m/03xp6r	"French" "AF"		special:N/N	((word-rel "special:N/N") "/m/0f8l9c" "/m/03xp6r")	[["special:N/N","/m/0f8l9c","/m/03xp6r"]]
/m/0f8l9c /m/03xp6r	"France" "Air Force"		special:N/N	((word-rel "special:N/N") "/m/0f8l9c" "/m/03xp6r")	[["special:N/N","/m/0f8l9c","/m/03xp6r"]]
"""

  val fileUtil = new FakeFileUtil
  fileUtil.addFileToBeRead(dataFile, dataFileContents)

  "readEntityNames" should "pull entity names out of a data file and sort them" in {
    val names = Evaluator.loadEntityNames(dataFile, fileUtil)
    names("/m/01cb_k") should be("\"H. pylori\"")
    names("/m/03z3fl") should be("\"J. Robin Warren\"")
    names("/m/0gkq30") should be("\"Car Service\"")
    names("/m/03rt9") should be("\"Ireland\"")
    names("/m/0f8l9c") should be("\"France\" \"French\"")
    names("/m/03xp6r") should be("\"AF\" \"Air Force\"")
  }


  "computeAveragePrecision" should "give the correct answer with a simple example" in {
    val scores = Seq((1.0, "a"), (.8, "b"), (.4, "c"))
    Evaluator.computeAveragePrecision(scores, Set("b")) should be(.5)
    Evaluator.computeAveragePrecision(scores, Set("b", "c")) should be(.5833 +- .0001)
    Evaluator.computeAveragePrecision(scores, Set("a", "b", "c")) should be(1)
    Evaluator.computeAveragePrecision(scores, Set("a", "b")) should be(1)
    Evaluator.computeAveragePrecision(scores, Set("a", "b", "d", "e", "f")) should be(.4 +- .0001)
  }

  it should "respect ties in scores, too" in {
    val scores = Seq((1.0, "a"), (1.0, "b"), (.8, "c"), (.8, "d"))
    Evaluator.computeAveragePrecision(scores, Set("a")) should be(.5 +- .0001)
    Evaluator.computeAveragePrecision(scores, Set("b")) should be(.5 +- .0001)
    Evaluator.computeAveragePrecision(scores, Set("c")) should be(.25 +- .0001)
    Evaluator.computeAveragePrecision(scores, Set("d")) should be(.25 +- .0001)
  }
}

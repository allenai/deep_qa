package org.allenai.semparse.evaluation

import org.scalatest._

import org.allenai.semparse.lisp.Environment

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

  "readEntityNames" should "pull entity names out of a data file" in {
    val env = new Environment(Seq(), Seq())
    val evaluator = new Evaluator(env, "", dataFile, fileUtil)
    evaluator.entityNames("/m/01cb_k") should be(Set("H. pylori"))
    evaluator.entityNames("/m/03z3fl") should be(Set("J. Robin Warren"))
    evaluator.entityNames("/m/0gkq30") should be(Set("Car Service"))
    evaluator.entityNames("/m/03rt9") should be(Set("Ireland"))
    evaluator.entityNames("/m/0f8l9c") should be(Set("French", "France"))
    evaluator.entityNames("/m/03xp6r") should be(Set("AF", "Air Force"))
  }
}

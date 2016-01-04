package org.allenai.semparse.one_off

import edu.cmu.ml.rtw.users.matt.util.FileUtil

import org.json4s._
import org.json4s.native.JsonMethods.parse

object get_mids_for_sfe {
  implicit val formats = DefaultFormats

  val training_file = "data/tacl2015-training-sample.txt"
  val test_file = "data/tacl2015-test.txt"

  val training_mids_file = "data/training-mids.txt"
  val test_mids_file = "data/test-mids.txt"
  val fileUtil = new FileUtil()

  def main(args: Array[String]) {
    val training_mids = get_training_mids()
    val test_mids = get_test_mids()
  }

  def get_training_mids() {
    // TODO(matt): separate mids and mid pairs into separate files, and make sure that mid pairs
    // are in sorted order.
    println(s"Getting training mids from $training_file")
    val mids = fileUtil.getLineIterator(training_file).map(_.split("\t")(0)).toSet.toSeq
    fileUtil.writeLinesToFile(training_mids_file, mids)
  }

  def get_test_mids() {
    // TODO(matt): separate mids and mid pairs into separate files, and make sure that mid pairs
    // are in sorted order.
    println(s"Getting test mids from $test_file")
    val mids = fileUtil.getLineIterator(test_file).toSeq.flatMap(line => {
      val json = parse(line)
      val queryMids = for {
        JObject(query) <- json
        JField("midsInQuery", JArray(mids)) <- query
      } yield mids.map(_.extract[String])
      queryMids.flatten
    }).toSet.toSeq
    fileUtil.writeLinesToFile(test_mids_file, mids)
  }
}

package org.allenai.semparse.one_off

import edu.cmu.ml.rtw.users.matt.util.FileUtil

import scala.collection.mutable

import org.json4s._
import org.json4s.native.JsonMethods.parse

object get_mids_for_sfe {
  implicit val formats = DefaultFormats

  val training_file = "data/tacl2015-training-sample.txt"
  val test_file = "data/tacl2015-test.txt"

  val training_mids_file = "data/training-mids.txt"
  val training_mid_pairs_file = "data/training-mid-pairs.txt"
  val test_mids_file = "data/test-mids.txt"
  val fileUtil = new FileUtil()

  def main(args: Array[String]) {
    val training_mids = get_training_mids()
    val test_mids = get_test_mids()
  }

  def get_training_mids() {
    println(s"Getting training mids from $training_file")
    val mids = new mutable.HashSet[String]
    val mid_pairs = new mutable.HashSet[(String, String)]
    for (line <- fileUtil.getLineIterator(training_file)) {
      val fields = line.split("\t")
      if (fields(0).contains(" ")) {
        // We're dealing with a mid pair, or a relation word, here.
        val mid_pair = fields(0).split(" ")
        mid_pairs += Tuple2(mid_pair(0), mid_pair(1))
        mids += mid_pair(0)
        mids += mid_pair(1)
        process_json_obj(parse(fields(5)), mids, mid_pairs)
      } else {
        // And here this is a mid, or a category word.
        val mid = fields(0)
        mids += mid
        process_json_obj(parse(fields(5)), mids, mid_pairs)
      }
    }
    fileUtil.writeLinesToFile(training_mids_file, mids.toSeq.sorted)
    fileUtil.writeLinesToFile(training_mid_pairs_file, mid_pairs.toSeq.sorted.map(x => x._1 + " " + x._2))
  }

  def process_json_obj(json_obj: JValue, mids: mutable.Set[String], mid_pairs: mutable.Set[(String, String)]) {
    val lists = for {
      JArray(list) <- json_obj
    } yield list
    // It's confusing to me that the (0) is necessary here, but it looks like it is...
    lists(0).map(list => {
      val strings = for {
        JString(string) <- list
      } yield string
      val mid_list = strings.drop(1)
      if (mid_list.size == 1) {
        mids += mid_list(0)
      } else if (mid_list.size == 2) {
        mid_pairs += Tuple2(mid_list(0), mid_list(1))
        mids += mid_list(0)
        mids += mid_list(1)
      } else {
        println("What happened here?")
        println(strings)
        println(mid_list)
        println(json_obj)
      }
    })
  }

  def get_test_mids() {
    // TODO(matt): This needs a lot of work.  For one, I don't think this does anything about mid
    // pairs, and we need to precompute the cross product of what we see here, I think.
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

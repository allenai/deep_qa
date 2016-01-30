package org.allenai.semparse.one_off

import edu.cmu.ml.rtw.users.matt.util.FileUtil

import scala.collection.mutable

import org.json4s._
import org.json4s.native.JsonMethods.parse

object get_mids_for_sfe {
  implicit val formats = DefaultFormats

  val training_file = "data/tacl2015-training.txt"
  val test_file = "data/tacl2015-test.txt"

  val training_mids_file = "data/large/training-mids.txt"
  val training_mid_pairs_file = "data/large/training-mid-pairs.txt"
  val mid_words_file = "data/large/training-mid-words.txt"
  val mid_pair_words_file = "data/large/training-mid-pair-words.txt"
  val test_mids_file = "data/large/test-mids.txt"
  val fileUtil = new FileUtil()

  def main(args: Array[String]) {
    get_training_mids()
    get_test_mids()
  }

  def get_training_mids() {
    println(s"Getting training mids from $training_file")
    val mids = new mutable.HashSet[String]
    val mid_pairs = new mutable.HashSet[(String, String)]
    val mid_words = new mutable.HashMap[String, Seq[String]].withDefaultValue(Nil)
    val mid_pair_words = new mutable.HashMap[(String, String), Seq[String]].withDefaultValue(Nil)
    var i = 0
    for (line <- fileUtil.getLineIterator(training_file)) {
      i += 1
      val fields = line.split("\t")
      if (fields(0).contains(" ")) {
        // We're dealing with a mid pair, or a relation word, here.
        val mid_pair_array = fields(0).split(" ")
        val word = fields(3)
        val mid_pair = Tuple2(mid_pair_array(0), mid_pair_array(1))
        mid_pairs += mid_pair
        mids += mid_pair._1
        mids += mid_pair._2
        process_json_obj(parse(fields(5)), mids, mid_pairs)
        mid_pair_words.update(mid_pair, mid_pair_words(mid_pair) :+ word)
      } else {
        // And here this is a mid, or a category word.
        val mid = fields(0)
        val word = fields(2)
        mids += mid
        try {
          process_json_obj(parse(fields(5)), mids, mid_pairs)
        } catch {
          case e: Exception => {
            println(s"Bad line $i: $line")
            println(s"Supposed json: ${fields(5)}")
            throw e
          }
        }
        mid_words.update(mid, mid_words(mid) :+ word)
      }
    }
    fileUtil.writeLinesToFile(training_mids_file, mids.toSeq.sorted)
    fileUtil.writeLinesToFile(training_mid_pairs_file, mid_pairs.toSeq.sorted.map(x => x._1 + " " + x._2))
    fileUtil.writeLinesToFile(mid_words_file, mid_words.map(mid_words_to_string).toSeq)
    fileUtil.writeLinesToFile(mid_pair_words_file, mid_pair_words.map(mid_pair_words_to_string).toSeq)
  }

  // TODO(matt): should I make these sets instead of seqs?  I guess the issue is, we're going to
  // use this to compute PMI.  Does the count for each word matter?  Maybe it does...  So I'm
  // keeping the seq for now.
  def mid_words_to_string(entry: (String, Seq[String])): String = {
    entry._1 + "\t" + entry._2.mkString("\t")
  }

  def mid_pair_words_to_string(entry: ((String, String), Seq[String])): String = {
    entry._1._1 + "," + entry._1._2 + "\t" + entry._2.mkString("\t")
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

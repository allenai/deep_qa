package org.allenai.semparse.one_off

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import edu.cmu.ml.rtw.users.matt.util.FileUtil

object SparkPmiComputer {

  // I guess I could make these command-line arguments, but for now, just change these and
  // recompile to run in different configurations.
  val dataSize = "large"  // "small" or "large"
  val runningLocally = true
  val midOrPairToRun = "mid pair"  // "mid" or "mid pair"

  val numPartitions = dataSize match {
    case "large" => 1000
    case "small" => 1500
  }

  // If we've seen a MID too many times, it blows up the word-feature computation.  The US, God,
  // and TV all show up more than 12k times in the large dataset.  I think we can get a good enough
  // PMI computation without these really frequent MIDs.
  // TODO(matt): these values are for "mid", not "mid pair"
  val MAX_WORD_COUNT = dataSize match {
    case "large" => 1000
    case "small" => 500
  }
  val MAX_FEATURE_COUNT = dataSize match {
    case "large" => 50000
    case "small" => 25000
  }

  val MIN_FEATURE_COUNT = dataSize match {
    case "large" => Map(("mid" -> 2000), ("mid pair" -> 100))
    case "small" => Map(("mid" -> 1000), ("mid pair" -> 50))
  }
  val FEATURES_PER_WORD = 100

  val matrixFile = if (runningLocally) {
    Map(
      ("mid" -> s"/home/mattg/pra/results/semparse/${dataSize}/mids/unknown/training_matrix.tsv"),
      ("mid pair" -> s"/home/mattg/pra/results/semparse/${dataSize}/mid_pairs/unknown/training_matrix.tsv")
    )
  } else {
    Map(
      ("mid" -> s"s3n://mattg-pipeline-tmp/${dataSize}_mid_training_matrix.tsv"),
      ("mid pair" -> s"s3n://mattg-pipeline-tmp/${dataSize}_mid_pair_training_matrix.tsv")
    )
  }

  val midWordFile = Map(
    ("mid" -> s"s3n://mattg-pipeline-tmp/${dataSize}_mid_words.tsv"),
    ("mid pair" -> s"s3n://mattg-pipeline-tmp/${dataSize}_mid_pair_words.tsv")
  )

  val wordFeatureFile = if (runningLocally) {
    Map(
      ("mid" -> s"/home/mattg/clone/tacl2015-factorization/data/${dataSize}/cat_word_features.tsv"),
      ("mid pair" -> s"/home/mattg/clone/tacl2015-factorization/data/${dataSize}/rel_word_features.tsv")
    )
  } else {
    Map(
      ("mid" -> s"s3n://mattg-pipeline-tmp/${dataSize}_cat_word_features.tsv"),
      ("mid pair" -> s"s3n://mattg-pipeline-tmp/${dataSize}_rel_word_features.tsv")
    )
  }

  def main(args: Array[String]) {
    // TODO(matt): make a better CLI, including getting the hard-coded parameters up top.
    val accessKeyId = args(0)
    val secretAccessKey = args(1)
    runSpark(midOrPairToRun, accessKeyId, secretAccessKey)
  }

  def runSpark(mid_or_pair: String, accessKeyId: String, secretAccessKey: String) {
    val conf = new SparkConf().setAppName(s"Compute PMI ($mid_or_pair)")
      .set("spark.driver.maxResultSize", "0")
      .set("fs.s3n.awsAccessKeyId", accessKeyId)
      .set("fs.s3n.awsSecretAccessKey", secretAccessKey)
      .set("spark.network.timeout", "1000000")
      .set("spark.akka.frameSize", "1028")

    if (runningLocally) {
      conf.setMaster("local[*]")
    }

    val sc = new SparkContext(conf)

    val minFeatureCount = MIN_FEATURE_COUNT(mid_or_pair)

    // PB here and below is "pre-broadcast" - it's a sign that you shouldn't be using this variable
    // anywhere except in a call to sc.broadcast()
    val midWordCountsPB = sc.textFile(midWordFile(mid_or_pair)).map(line => {
      val fields = line.split("\t")
      val mid = fields(0).replace(" ", ",")
      val words = fields.drop(1).toSeq
      // The last .map(identity) is because of a scala bug - Map#mapValues is not serializable.
      val wordCounts = words.groupBy(identity).mapValues(_.size).map(identity)
      (mid -> wordCounts.toMap)
    }).filter(_._2.map(_._2).foldLeft(0)(_+_) < MAX_WORD_COUNT)
    println(s"Read words for ${midWordCountsPB.count} mids")
    val midWordCounts = sc.broadcast(midWordCountsPB.collectAsMap)

    val featureCountsPB = sc.textFile(matrixFile(mid_or_pair), numPartitions).flatMap(line => {
      val fields = line.split("\t")
      val mid = fields(0)
      val words = midWordCounts.value.getOrElse(mid, Seq.empty)
      val features = fields(2).trim.split(" -#- ")
      if (features.length < MAX_FEATURE_COUNT) {
        val totalCount = words.map(_._2).foldLeft(0)(_+_).toDouble
        if (totalCount > 0) {
          for (feature <- features.map(f => f.replace(",1.0", ""))) yield (feature -> totalCount)
        } else {
          Seq()
        }
      } else {
        Seq()
      }
    }).reduceByKey(_+_).filter(_._2 > minFeatureCount)
    val featureCounts = sc.broadcast(featureCountsPB.collectAsMap)

    val wordFeatureCounts = sc.textFile(matrixFile(mid_or_pair), numPartitions).flatMap(line => {
      val fields = line.split("\t")
      val mid = fields(0)
      val words = midWordCounts.value.getOrElse(mid, Seq.empty)
      val features = fields(2).trim.split(" -#- ")
      if (features.length < MAX_FEATURE_COUNT) {
        val counts = featureCounts.value
        for ((word, wordCount) <- words;
             feature <- features.map(f => f.replace(",1.0", ""));
             if counts.getOrElse(feature, 0.0) > minFeatureCount) yield ((word, feature) -> wordCount.toDouble)
      } else {
        Seq()
      }
    }).reduceByKey(_+_)
    wordFeatureCounts.persist()

    val wordCountsPB = sc.textFile(matrixFile(mid_or_pair), numPartitions).flatMap(line => {
      val fields = line.split("\t")
      val mid = fields(0)
      val words = midWordCounts.value.getOrElse(mid, Seq.empty)
      val features = fields(2).trim.split(" -#- ")
      val numFeatures = features.length.toDouble
      if (numFeatures < MAX_FEATURE_COUNT) {
        for ((word, wordCount) <- words) yield (word -> wordCount * numFeatures)
      } else {
        Seq()
      }
    }).reduceByKey(_+_)
    val wordCounts = sc.broadcast(wordCountsPB.collectAsMap)

    val totalCountPB = wordCountsPB.map(_._2).reduce(_+_)
    val totalCount = sc.broadcast(totalCountPB.toDouble)

    val pmiScores = wordFeatureCounts.map(entry => {
      val ((word, feature), wordFeatureCount) = entry
      val wordCount = wordCounts.value(word)
      val featureCount = featureCounts.value.getOrElse(feature, 0.0)
      val pmi = if (featureCount < minFeatureCount) {
        0
      } else {
        (totalCount.value * wordFeatureCount) / (wordCount * featureCount)
      }
      (word, (feature, pmi))
    })

    val pmiFailures = sc.accumulator(0, "PMI failures")
    val keptFeatures = pmiScores.aggregateByKey(List[(String, Double)]())((partialList, item) => item :: partialList, (list1, list2) => list1 ::: list2)
      .map(entry => {
        try {
          val (word, featureScores) = entry
          val grouped = featureScores.groupBy(_._2).toSeq.sortBy(-_._1).take(FEATURES_PER_WORD)
          val kept = grouped.flatMap(entry => {
            val score = entry._1
            if (score > 0.0) {
              val features = entry._2.map(_._1)
              val shortest_feature = features.sortBy(_.length).head
              Seq((shortest_feature, score))
            } else {
              Seq()
            }
          })
          (word, kept)
        } catch {
          case e: Exception => {
            pmiFailures += 1
            (entry._1, Seq(("ERROR IN PMI COMPUTATION!", 0.0)))
          }
        }
      })

    val records = keptFeatures.map(entry => {
      val (word, features) = entry
      // Always include CONNECTED and bias as potential features.  bias will get automatically
      // added to every entity / entity pair, and CONNECTED means that two entities have a direct
      // connection in the graph.
      val defaultFeatures = if (mid_or_pair == "mid pair") Set("bias", "CONNECTED") else Set("bias")
      val f = features.map(_._1).toSet ++ defaultFeatures
      val featureStr = f.mkString("\t")
      s"$word\t$featureStr"
    })
    val results = records.collect()
    if (runningLocally) {
      new FileUtil().writeLinesToFile(wordFeatureFile(mid_or_pair), results)
    } else {
      sc.parallelize(results, 1).saveAsTextFile(wordFeatureFile(mid_or_pair))
    }
  }
}

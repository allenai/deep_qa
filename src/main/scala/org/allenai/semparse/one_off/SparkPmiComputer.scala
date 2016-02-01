package org.allenai.semparse.one_off

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object SparkPmiComputer {

  val dataSize = "small"
  val MIN_FEATURE_COUNT = dataSize match {
    case "large" => Map(("mid" -> 2000), ("mid pair" -> 100))
    case "small" => Map(("mid" -> 1000), ("mid pair" -> 50))
  }
  val FEATURES_PER_WORD = 100

  val matrixFile = Map(
    ("mid" -> s"s3://mattg-pipeline-tmp/${dataSize}_mid_training_matrix.tsv"),
    ("mid pair" -> s"s3://mattg-pipeline-tmp/${dataSize}_mid_pair_training_matrix.tsv")
  )

  val midWordFile = Map(
    ("mid" -> s"s3://mattg-pipeline-tmp/${dataSize}_mid_words.tsv"),
    ("mid pair" -> s"s3://mattg-pipeline-tmp/${dataSize}_mid_pair_words.tsv")
  )

  val wordFeatureFile = Map(
    ("mid" -> s"s3://mattg-pipeline-tmp/${dataSize}_cat_word_features.tsv"),
    ("mid pair" -> s"s3://mattg-pipeline-tmp/${dataSize}_rel_word_features.tsv")
  )

  def main(args: Array[String]) {
    runSpark("mid")
    //runSpark("mid pair")
  }

  def runSpark(mid_or_pair: String) {
    val conf = new SparkConf().setAppName(s"Compute PMI ($mid_or_pair)")
    val sc = new SparkContext(conf)

    val minFeatureCount = MIN_FEATURE_COUNT(mid_or_pair)

    // PB here and below is "pre-broadcast" - it's a sign that you shouldn't be using this variable
    // anywhere except in a call to sc.broadcast()
    val midWordCountsPB = sc.textFile(midWordFile(mid_or_pair)).map(line => {
      val fields = line.split("\t")
      val mid = fields(0)
      val words = fields.drop(1).toSeq
      // The last .map(identity) is because of a scala bug - Map#mapValues is not serializable.
      val wordCounts = words.groupBy(identity).mapValues(_.size).map(identity)
      (mid -> wordCounts.toMap)
    })
    println(s"Read words for ${midWordCountsPB.count} mids")
    val midWordCounts = sc.broadcast(midWordCountsPB.collectAsMap)

    val wordFeatureCounts = sc.textFile(matrixFile(mid_or_pair), 1500).flatMap(line => {
      val fields = line.split("\t")
      val mid = fields(0)
      val words = midWordCounts.value.getOrElse(mid, Seq.empty)
      val features = fields(2).trim.split(" -#- ").map(f => f.replace(",1.0", ""))
      for ((word, wordCount) <- words;
           feature <- features) yield ((word, feature) -> wordCount)
    }).reduceByKey(_+_)
    wordFeatureCounts.persist()

    val wordCountsPB = wordFeatureCounts.map(entry => (entry._1._1, entry._2)).reduceByKey(_+_)
    wordCountsPB.persist()
    val wordCounts = sc.broadcast(wordCountsPB.collectAsMap)

    val totalCountPB = wordCountsPB.map(_._2).reduce(_+_)
    val totalCount = sc.broadcast(totalCountPB.toDouble)

    val featureCountsPB = wordFeatureCounts.map(entry => (entry._1._2, entry._2)).reduceByKey(_+_)
    val featureCounts = sc.broadcast(featureCountsPB.collectAsMap)


    val pmiScores = wordFeatureCounts.map(entry => {
      val ((word, feature), wordFeatureCount) = entry
      val wordCount = wordCounts.value(word)
      val featureCount = featureCounts.value(feature)
      val pmi = if (featureCount < minFeatureCount) {
        0
      } else {
        (totalCount.value * wordFeatureCount) / (wordCount * featureCount)
      }
      (word, (feature, pmi))
    })
    val keptFeatures = pmiScores.aggregateByKey(List[(String, Double)]())((partialList, item) => item :: partialList, (list1, list2) => list1 ::: list2)
      .map(entry => {
        val (word, featureScores) = entry
        val grouped = featureScores.groupBy(_._2).toSeq.sortBy(-_._1)
        val kept = grouped.flatMap(entry => {
          val score = entry._1
          if (score > 0.0) {
            val features = entry._2.map(_._1)
            val shortest_feature = features.sortBy(_.length).head
            println(s"word: $word, shortest feature: $shortest_feature, score: $score")
            Seq((shortest_feature, score))
          } else {
            Seq()
          }
        })
        (word, kept)
      })
    keptFeatures.coalesce(1, false).saveAsTextFile(wordFeatureFile(mid_or_pair))
  }
}

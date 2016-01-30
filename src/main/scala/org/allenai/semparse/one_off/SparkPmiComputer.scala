package org.allenai.semparse.one_off

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object SparkPmiComputer {

  val dataSize = "small"
  val MIN_FEATURE_COUNT = dataSize match {
    case "large" => Map(("mid" -> 5000), ("mid pair" -> 500))
    case "small" => Map(("mid" -> 1000), ("mid pair" -> 50))
  }
  val FEATURES_PER_WORD = 100

  val matrixFile = Map(
    ("mid" -> s"/home/mattg/pra/results/semparse/$dataSize/mids/unknown/training_matrix.tsv"),
    ("mid pair" -> s"/home/mattg/pra/results/semparse/$dataSize/mid_pairs/unknown/training_matrix.tsv")
  )

  val midWordFile = Map(
    ("mid" -> s"/home/mattg/clone/tacl2015-factorization/data/$dataSize/training-mid-words.txt"),
    ("mid pair" -> s"/home/mattg/clone/tacl2015-factorization/data/$dataSize/training-mid-pair-words.txt")
  )

  /*
  val wordFeatureFile = Map(
    ("mid" -> s"/home/mattg/clone/tacl2015-factorization/data/$dataSize/cat_word_features.tsv"),
    ("mid pair" -> s"/home/mattg/clone/tacl2015-factorization/data/$dataSize/rel_word_features.tsv")
  )
  */

  val wordFeatureFile = Map(
    ("mid" -> s"/home/mattg/clone/tacl2015-factorization/test_cat_word_features.tsv"),
    ("mid pair" -> s"/home/mattg/clone/tacl2015-factorization/test_rel_word_features.tsv")
  )

  def main(args: Array[String]) {
    runSpark("mid")
    //runSpark("mid pair")
  }

  def runSpark(mid_or_pair: String) {
    val conf = new SparkConf().setAppName(s"Compute PMI ($mid_or_pair)").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val minFeatureCount = MIN_FEATURE_COUNT(mid_or_pair)

    // PB here and below is "pre-broadcast" - it's a sign that you shouldn't be using this variable
    // anywhere except in a call to sc.broadcast()
    val midWordsPB = sc.textFile(midWordFile(mid_or_pair)).map(line => {
      val fields = line.split("\t")
      val mid = fields(0)
      val words = fields.drop(1).toSeq
      (mid -> words)
    })
    println(s"Read words for ${midWordsPB.count} mids")
    val midWords = sc.broadcast(midWordsPB.collectAsMap)

    val wordFeatureCounts = sc.textFile(matrixFile(mid_or_pair), 1500).flatMap(line => {
      val fields = line.split("\t")
      val mid = fields(0)
      val words = midWords.value.getOrElse(mid, Seq.empty)
      // TODO(matt): make these counts first, it should cut down a _lot_ on the intermediate output
      val features = fields(2).trim.split(" -#- ").map(f => f.replace(",1.0", ""))
      for (word <- words;
           feature <- features) yield ((word, feature) -> 1)
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

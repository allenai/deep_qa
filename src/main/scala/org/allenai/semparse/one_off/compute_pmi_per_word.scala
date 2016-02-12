package org.allenai.semparse.one_off

import edu.cmu.ml.rtw.users.matt.util.FileUtil

import sys.process._

import java.io.FileWriter

import scala.collection.mutable

// Looks like this is too large to do in memory, so we're going to do a pipeline approach.
object compute_pmi_per_word {
  val fileUtil = new FileUtil

  val dataSize = "small"

  val matrix_file = Map(
    ("mid" -> s"/home/mattg/pra/results/semparse/$dataSize/mids/unknown/training_matrix.tsv"),
    ("mid pair" -> s"/home/mattg/pra/results/semparse/$dataSize/mid_pairs/unknown/training_matrix.tsv")
  )
  val mid_word_file = Map(
    ("mid" -> s"/home/mattg/clone/tacl2015-factorization/data/$dataSize/training-mid-words.txt"),
    ("mid pair" -> s"/home/mattg/clone/tacl2015-factorization/data/$dataSize/training-mid-pair-words.txt")
  )
  val word_feature_file = Map(
    ("mid" -> s"/home/mattg/clone/tacl2015-factorization/data/$dataSize/cat_word_features.tsv"),
    ("mid pair" -> s"/home/mattg/clone/tacl2015-factorization/data/$dataSize/rel_word_features.tsv")
  )
  val filtered_feature_file = Map(
    ("mid" -> s"/home/mattg/clone/tacl2015-factorization/data/$dataSize/mid_features.tsv"),
    ("mid pair" -> s"/home/mattg/clone/tacl2015-factorization/data/$dataSize/mid_pair_features.tsv")
  )
  val tmp_dir = Map(
    ("mid" -> s"tmp_mid/$dataSize/"),
    ("mid pair" -> s"tmp_mid_pair/$dataSize/")
  )

  val MIN_FEATURE_COUNT = dataSize match {
    case "large" => Map(("mid" -> 5000), ("mid pair" -> 500))
    case "small" => Map(("mid" -> 1000), ("mid pair" -> 50))
  }
  val FEATURES_PER_WORD = 100

  fileUtil.mkdirs(tmp_dir("mid"))
  fileUtil.mkdirs(tmp_dir("mid pair"))

  val emptyLogger = ProcessLogger(line => ())

  def main(args: Array[String]) {
    runPipeline("mid")
    runPipeline("mid pair")
  }

  def runPipeline(mid_or_pair: String) {
    println(s"Running pipeline for $mid_or_pair")
    // First we take the original matrix and find a set of potential features.
    //val features = getFeaturesThatMeetCutoff(matrix_file(mid_or_pair), MIN_FEATURE_COUNT(mid_or_pair))
    //println(s"Kept ${features.size} features")

    // Then we go through the matrix file again, and output new files that we can use to get the
    // counts we need for computing PMI.
    //createCountFiles(mid_or_pair, features)

    // Then we actually read those count files and do the PMI computation.
    //loadFilesAndComputePmi(mid_or_pair)

    // Finally, we go through the feature matrix we output in createCountFiles and remove all
    // features that weren't selected by the PMI computation.
    filterFeatureFile(mid_or_pair)
  }

  def filterFeatureFile(mid_or_pair: String) {
    println(s"Filtering features for ${mid_or_pair}s")
    println("Reading kept word features")
    val keptFeatures = fileUtil.getLineIterator(word_feature_file(mid_or_pair)).flatMap(line => {
      line.split("\t").drop(1).toSeq
    }).toSet
    println("Reading the feature matrix")
    val filteredFeatures = do_feature_selection.readFeaturesFromFile(matrix_file(mid_or_pair), keptFeatures)
    println("Outputting a final feature matrix")
    do_feature_selection.outputFeatureMatrix(filteredFeatures, filtered_feature_file(mid_or_pair))
  }

  def getFeaturesThatMeetCutoff(filename: String, min_feature_count: Int): Set[String] = {
    println(s"Getting global feature counts from $filename")
    val featuresFromLine: String => Seq[String] = line => {
      line.split("\t")(2).trim.split(" -#- ").map(f => f.replace(",1.0", ""))
    }
    val counts = fileUtil.getCountsFromFile(filename, featuresFromLine)
    println(s"Found ${counts.size} total features")
    println(s"Filtering features, keeping only those with count of at least $min_feature_count")
    counts.par.filter(entry => entry._2 >= min_feature_count).keys.seq.toSet
  }

  def loadFeatureCounts(filename: String, min_feature_count: Int): Map[String, Int] = {
    println("Loading feature counts")
    fileUtil.getLineIterator(filename).flatMap(line => {
      val fields = line.split("\t")
      val count = fields(1).toInt
      if (count > min_feature_count) {
        val feature = fields(0).substring(8, fields(0).length - 1)
        Seq((feature -> count))
      } else {
        Seq()
      }
    }).toMap
  }

  def loadWordCounts(filename: String): Map[String, Int] = {
    println("Loading word counts")
    fileUtil.getLineIterator(filename).map(line => {
      val fields = line.split("\t")
      val word = fields(0).substring(5, fields(0).length - 1)
      val count = fields(1).toInt
      (word -> count)
    }).toMap
  }

  def loadWordFeatureCounts(filename: String, features: Set[String]): (Map[(String, String), Int], Double) = {
    println("Loading word feature counts")
    println(s"There are ${features.size} possible features")
    var totalCount = 0.0
    var i = 0
    val wordFeatures = fileUtil.getLineIterator(filename).flatMap(line => {
      fileUtil.logEvery(1000000, i)
      val fields = line.split("\t")
      val feature = fields(1).substring(8, fields(1).length - 1)
      val count = fields(2).toInt
      totalCount += count
      i += 1
      if (features.contains(feature)) {
        val word = fields(0).substring(5, fields(0).length - 1)
        Seq(((word, feature) -> count))
      } else {
        Seq()
      }
    }).toMap
    (wordFeatures, totalCount)
  }

  def createCountFiles(mid_or_pair: String, allowedFeatures: Set[String]) {
    println("Loading results file")
    val featuresForMid =
      do_feature_selection.readFeaturesFromFile(matrix_file(mid_or_pair), allowedFeatures)
    println(s"Found features for ${featuresForMid.size} MIDs")

    println("Reading MID word file")
    val midWords = do_feature_selection.readMidWords(mid_word_file(mid_or_pair))
    println(s"Found words for ${midWords.size} MIDs")

    println("Doing initial map from mids to words")
    val wordFeatures = getWordFeatureList(featuresForMid, midWords)

    val totalCount = wordFeatures.size

    writeAndSortWordFeatureOccurrences(tmp_dir(mid_or_pair), wordFeatures)
    writeAndSortWordOccurrences(tmp_dir(mid_or_pair), wordFeatures)
    writeAndSortFeatureOccurrences(tmp_dir(mid_or_pair), wordFeatures)
  }

  def writeAndSortWordFeatureOccurrences(base: String, wordFeatures: Seq[(Word, Feature)]) {
    println("Outputting a file with word feature occurrences")
    fileUtil.writeLinesToFile(base + "word_feature_occurrences.txt", wordFeatures.view.map(x => s"${x._1}\t${x._2}"))
    println("Sorting and counting them")
    s"sort -S 100g --parallel=40 ${base}word_feature_occurrences.txt" #| "uniq -c" #> new java.io.File(base + "word_feature_counts_bad_format.txt") ! emptyLogger
    s"rm -f ${base}word_feature_occurrences.txt" ! emptyLogger
    println("Fixing the file format")
    Seq("awk", "{print $2\"\t\"$3\"\t\"$1}", s"${base}word_feature_counts_bad_format.txt") #> new java.io.File(base + "word_feature_counts.txt") ! emptyLogger
    s"rm -f ${base}word_feature_counts_bad_format.txt" ! emptyLogger
  }

  def writeAndSortWordOccurrences(base: String, wordFeatures: Seq[(Word, Feature)]) {
    println("Outputting a file with word occurrences")
    fileUtil.writeLinesToFile(base + "word_occurrences.txt", wordFeatures.view.map(x => s"${x._1}"))
    println("Sorting and counting them")
    s"sort -S 100g --parallel=40 ${base}word_occurrences.txt" #| "uniq -c" #> new java.io.File(base + "word_counts_bad_format.txt") ! emptyLogger
    s"rm -f ${base}word_occurrences.txt" ! emptyLogger
    println("Fixing the file format")
    Seq("awk", "{print $2\"\t\"$1}", s"${base}word_counts_bad_format.txt") #> new java.io.File(base + "word_counts.txt") ! emptyLogger
    s"rm -f ${base}word_counts_bad_format.txt" ! emptyLogger
  }

  def writeAndSortFeatureOccurrences(base: String, wordFeatures: Seq[(Word, Feature)]) {
    println("Outputting a file with feature occurrences")
    fileUtil.writeLinesToFile(base + "feature_occurrences.txt", wordFeatures.view.map(x => s"${x._2}"))
    println("Sorting and counting them")
    s"sort -S 100g --parallel=40 ${base}feature_occurrences.txt" #| "uniq -c" #> new java.io.File(base + "feature_counts_bad_format.txt") ! emptyLogger
    s"rm -f ${base}feature_occurrences.txt" ! emptyLogger
    println("Fixing the file format")
    Seq("awk", "{print $2\"\t\"$1}", s"${base}feature_counts_bad_format.txt") #> new java.io.File(base + "feature_counts.txt") ! emptyLogger
    s"rm -f ${base}feature_counts_bad_format.txt" ! emptyLogger
  }

  def getWordFeatureList(featuresForMid: Map[Mid, Seq[Feature]], midWords: Map[Mid, Seq[Word]]) = {
    featuresForMid.toSeq.par.flatMap(entry => {
      val mid = entry._1
      val features = entry._2
      midWords.getOrElse(mid, Seq.empty).flatMap(word => {
        features.map(feature => {
          (word, feature)
        })
      })
    }).seq
  }

  def loadFilesAndComputePmi(mid_or_pair: String) {
    val base = tmp_dir(mid_or_pair)
    val featureCounts = loadFeatureCounts(base + "feature_counts.txt", MIN_FEATURE_COUNT(mid_or_pair))
    val (wordFeatureCounts, totalCount) = loadWordFeatureCounts(base + "word_feature_counts.txt", featureCounts.keySet)
    val wordCounts = loadWordCounts(base + "word_counts.txt")
    println("Loaded all files, computing PMI")
    val wordFeatures = wordCounts.par.map(entry => {
      val word = entry._1
      val wordCount = entry._2
      val features = selectWordFeatures(word, wordCount, totalCount, wordFeatureCounts, featureCounts)
      (word, features)
    })
    println("Writing results to disk")
    writeWordFeaturesToDisk(word_feature_file(mid_or_pair), wordFeatures.seq)
  }

  def selectWordFeatures(
    word: String,
    wordCount: Int,
    totalCount: Double,
    wordFeatureCounts: Map[(String, String), Int],
    featureCounts: Map[String, Int]
  ): Seq[(String, Double)] = {
    val scoredFeatures = featureCounts.map(entry => {
      val feature = entry._1
      val featureCount = entry._2
      val wordFeatureCount = wordFeatureCounts.getOrElse((word, feature), 0)
      (feature, (totalCount * wordFeatureCount) / (wordCount * featureCount))
    })
    val grouped = scoredFeatures.toSeq.groupBy(_._2).toSeq.sortBy(-_._1)
    val kept = grouped.flatMap(entry => {
      val score = entry._1
      if (score > 0.0) {
        val features = entry._2.map(_._1)
        val shortest_feature = features.sortBy(_.length).head
        Seq((shortest_feature, score))
      } else {
        Seq()
      }
    }).take(FEATURES_PER_WORD)
    kept
  }

  def writeWordFeaturesToDisk(filename: String, wordFeatures: Iterable[(String, Seq[(String, Double)])]) {
    val writer = fileUtil.getFileWriter(filename)
    for (wordAndFeatures <- wordFeatures) {
      val word = wordAndFeatures._1
      val features = wordAndFeatures._2
      writeWordFeaturesToFile(word, features, writer)
    }
    writer.close()
  }

  def writeWordFeaturesToFile(
    word: String,
    features: Seq[(String, Double)],
    writer: FileWriter,
    humanReadable: Boolean = false
  ) {
    if (humanReadable) {
      writer.write(word)
      writer.write("\n")
      for (featureScore <- features) {
        writer.write(s"   ${featureScore._1}\t${featureScore._2}\n")
      }
    } else {
      writer.write(word)
      for (featureScore <- features) {
        writer.write(s"\t${featureScore._1}")
      }
      writer.write("\tbias\n")
    }
  }

}

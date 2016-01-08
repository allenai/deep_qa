package org.allenai.semparse.one_off

import edu.cmu.ml.rtw.users.matt.util.FileUtil

import sys.process._

// Looks like this is too large to do in memory, so we're going to do a pipeline approach.
object compute_pmi_per_word {
  val fileUtil = new FileUtil

  val mid_results_file = "/home/mattg/pra/results/semparse/mids/unknown/training_matrix.tsv"
  val mid_word_file = "/home/mattg/clone/tacl2015-factorization/data/training-mid-words.txt"
  val tmp_dir = "tmp/"

  fileUtil.mkdirs(tmp_dir)

  val emptyLogger = ProcessLogger(line => ())

  def main(args: Array[String]) {
    //createCountFiles()
  }

  def createCountFiles() {
    println("Loading results file")
    val featuresForMid = do_feature_selection.readFeaturesFromFile(mid_results_file)

    println("Reading MID word file")
    val midWords = do_feature_selection.readMidWords(mid_word_file)

    println("Doing initial map from mids to words")
    val wordFeatures = getWordFeatureList(featuresForMid, midWords)

    val totalCount = wordFeatures.size

    writeAndSortWordFeatureOccurrences(wordFeatures)
    writeAndSortWordOccurrences(wordFeatures)
    writeAndSortFeatureOccurrences(wordFeatures)
  }

  def writeAndSortWordFeatureOccurrences(wordFeatures: Seq[(Word, Feature)]) {
    println("Outputting a file with word feature occurrences")
    fileUtil.writeLinesToFile("tmp/word_feature_occurrences.txt", wordFeatures.view.map(x => s"${x._1}\t${x._2}"))
    println("Sorting and counting them")
    "sort -S 100g --parallel=40 tmp/word_feature_occurrences.txt" #| "uniq -c" #> new java.io.File("tmp/word_feature_counts_bad_format.txt") ! emptyLogger
    "rm -f tmp/word_feature_occurrences.txt" ! emptyLogger
    println("Fixing the file format")
    Seq("awk", "{print $2\"\t\"$3\"\t\"$1}", "tmp/word_feature_counts_bad_format.txt") #> new java.io.File("tmp/word_feature_counts.txt") ! emptyLogger
    "rm -f tmp/word_feature_counts_bad_format.txt" ! emptyLogger
  }

  def writeAndSortWordOccurrences(wordFeatures: Seq[(Word, Feature)]) {
    println("Outputting a file with word occurrences")
    fileUtil.writeLinesToFile("tmp/word_occurrences.txt", wordFeatures.view.map(x => s"${x._1}"))
    println("Sorting and counting them")
    "sort -S 100g --parallel=40 tmp/word_occurrences.txt" #| "uniq -c" #> new java.io.File("tmp/word_counts_bad_format.txt") ! emptyLogger
    "rm -f tmp/word_occurrences.txt" ! emptyLogger
    println("Fixing the file format")
    Seq("awk", "{print $2\"\t\"$1}", "tmp/word_counts_bad_format.txt") #> new java.io.File("tmp/word_counts.txt") ! emptyLogger
    "rm -f tmp/word_counts_bad_format.txt" ! emptyLogger
  }

  def writeAndSortFeatureOccurrences(wordFeatures: Seq[(Word, Feature)]) {
    println("Outputting a file with feature occurrences")
    fileUtil.writeLinesToFile("tmp/feature_occurrences.txt", wordFeatures.view.map(x => s"${x._2}"))
    println("Sorting and counting them")
    "sort -S 100g --parallel=40 tmp/feature_occurrences.txt" #| "uniq -c" #> new java.io.File("tmp/feature_counts_bad_format.txt") ! emptyLogger
    "rm -f tmp/feature_occurrences.txt" ! emptyLogger
    println("Fixing the file format")
    Seq("awk", "{print $2\"\t\"$1}", "tmp/feature_counts_bad_format.txt") #> new java.io.File("tmp/feature_counts.txt") ! emptyLogger
    "rm -f tmp/feature_counts_bad_format.txt" ! emptyLogger
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
}

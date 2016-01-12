package org.allenai.semparse.one_off

import edu.cmu.ml.rtw.users.matt.util.FileUtil

import scala.collection.mutable

import sys.process._

// These are just so the typing is nicer below, to help me keep things straight.
case class Mid(mid: String)  // note: this could also be a mid pair
case class Word(word: String)
case class Feature(feature: String)

class Counts(
  wordFeatureCounts: Map[(Word, Feature), Int],
  featureCounts: Map[Feature, Int],
  wordCounts: Map[Word, Int],
  totalCount: Int
) {

  /*
  def observeFeaturesWithWord(word: Word, features: Seq[Feature]) {
    for (feature <- features) observeFeatureWithWord(word, feature)
  }

  def observeFeatureWithWord(word: Word, feature: Feature) {
    totalCount += 1
    wordFeatureCounts.update((word, feature), wordFeatureCounts((word, feature)) + 1)
    wordCounts.update(word, wordCounts(word) + 1)
    featureCounts.update(feature, featureCounts(feature) + 1)
  }
  */

  def allFeatures = featureCounts.keys
  def allWords = wordCounts.keys

  def getFeatureCount(feature: Feature) = featureCounts(feature)
  def getWordCount(word: Word) = wordCounts(word)
  def getWordFeatureCount(word: Word, feature: Feature) = wordFeatureCounts((word, feature))
}

object do_feature_selection {
  val fileUtil = new FileUtil

  def main(args: Array[String]) {
    println("Selecting MID features")
    featureSelectionFromFile(
      "/home/mattg/pra/results/semparse/mids/unknown/training_matrix.tsv",
      "/home/mattg/clone/tacl2015-factorization/data/training-mid-words.txt",
      "word-graph-features",
      "/home/mattg/clone/tacl2015-factorization/data/mid_features"
    )
    println("Selecting MID pair features")
    featureSelectionFromFile(
      "/home/mattg/pra/results/semparse/mid_pairs/unknown/training_matrix.tsv",
      "/home/mattg/clone/tacl2015-factorization/data/training-mid-pair-words.txt",
      "word-rel-graph-features",
      "/home/mattg/clone/tacl2015-factorization/data/mid_pair_features"
    )
  }

  def featureSelectionFromFile(
    infile: String,
    mid_word_file: String,
    dictionaryName: String,
    outfile: String
  ) {
    println("Reading features from file")
    val featuresForMid = readFeaturesFromFile(infile)
    printTopKeys(featuresForMid, 10)
    println("Reading MID-word file")
    val midWords = readMidWords(mid_word_file)
    println("Getting feature counts by word")
    val counts = getFeatureCounts(featuresForMid, midWords)
    println("Selecting features")
    val keptFeaturesForWords = selectFeaturesForWords(counts)
    val keptFeatures = keptFeaturesForWords.flatMap(_._2).toSet
    println(s"Kept ${keptFeatures.size} features")
    println("Filtering features")
    val filteredFeatures = filterFeatures(featuresForMid, keptFeatures)
    printTopKeys(filteredFeatures, 10)
    println("Outputting feature matrix")
    outputFeatureMatrix(filteredFeatures, outfile + ".tsv")
    println("Outputting feature dictionary")
    outputFeatureDictionary(filteredFeatures, dictionaryName, outfile + "_list.txt")
    println("Outputting word features")
    outputWordFeatures(keptFeaturesForWords, outfile + "_per_word.tsv")
  }

  def readFeaturesFromFile(infile: String): Map[Mid, Seq[Feature]] = {
    println(s"Reading features from file $infile")
    fileUtil.getLineIterator(infile).grouped(1024).flatMap(lines => {
      lines.par.map(line => {
        val fields = line.split("\t")
        val key = fields(0).trim.replace(",", " ")
        val features = fields(2).trim.split(" -#- ").map(f => Feature(f.replace(",1.0", "")))
        (Mid(key), features.toSeq)
      })
    }).toMap.seq
  }

  def readMidWords(infile: String): Map[Mid, Seq[Word]] = {
    fileUtil.readMapListFromTsvFile(infile).map(entry => {
      val mid = Mid(entry._1)
      val words = entry._2.map(x => Word(x))
      (mid, words)
    })
  }

  def getFeatureCounts(
    featuresForMid: Map[Mid, Seq[Feature]],
    midWords: Map[Mid, Seq[Word]]
  ): Counts = {
    println("Doing initial map from mids to words")
    val wordFeatures = featuresForMid.toSeq.par.flatMap(entry => {
      val mid = entry._1
      val features = entry._2
      midWords.getOrElse(mid, Seq.empty).flatMap(word => {
        features.map(feature => {
          (word, feature)
        })
      })
    }).seq

    val totalCount = wordFeatures.size

    println("Outputting a file with word feature occurrences")
    fileUtil.writeLinesToFile("tmp_word_feature_occurrences.txt", wordFeatures.view.map(x => s"${x._1}\t${x._2}"))
    "sort tmp_word_feature_occurrences.txt" #| "uniq -c" #> new java.io.File("tmp_word_feature_counts.txt") ! ProcessLogger(line => ())
    println("Reading sorted file and getting counts")
    val wordFeatureCounts = fileUtil.readLinesFromFile("tmp_word_feature_counts.txt").map(line => {
      val fields = line.trim.split(" ")
      val count = fields(0).toInt
      val (word, feature) = fields(1).splitAt(fields(1).indexOf("\t"))
      ((Word(word.trim), Feature(feature.trim)), count)
    }).toMap

    println("Outputting a file with word occurrences")
    fileUtil.writeLinesToFile("tmp_word_occurrences.txt", wordFeatures.view.map(x => s"${x._1}"))
    "sort tmp_word_occurrences.txt" #| "uniq -c" #> new java.io.File("tmp_word_counts.txt") ! ProcessLogger(line => ())
    println("Reading sorted file and getting counts")
    val wordCounts = fileUtil.readLinesFromFile("tmp_word_counts.txt").map(line => {
      val fields = line.trim.split(" ")
      val count = fields(0).toInt
      val word = fields(1)
      (Word(word), count)
    }).toMap


    println("Outputting a file with feature occurrences")
    fileUtil.writeLinesToFile("tmp_feature_occurrences.txt", wordFeatures.view.map(x => s"${x._2}"))
    "sort tmp_feature_occurrences.txt" #| "uniq -c" #> new java.io.File("tmp_feature_counts.txt") ! ProcessLogger(line => ())
    println("Reading sorted file and getting counts")
    val featureCounts = fileUtil.readLinesFromFile("tmp_feature_counts.txt").map(line => {
      val fields = line.trim.split(" ")
      val count = fields(0).toInt
      val feature = fields(1)
      (Feature(feature), count)
    }).toMap

    new Counts(wordFeatureCounts, featureCounts, wordCounts, totalCount)
  }

  def printTopKeys(featuresForMid: Map[Mid, Seq[Feature]], topK: Int) {
    val topMids = featuresForMid.keys.toSeq.sortBy(mid => -featuresForMid(mid).size).take(topK)
    println("Top mids:")
    for (mid <- topMids) {
      println(s"${mid.mid} -> ${featuresForMid(mid).size}")
    }
  }

  def printFeatureCountHistogram(featureCounts: Map[Feature, Int]) {
    val featureCountHistogram = new mutable.HashMap[Int, Int].withDefaultValue(0)
    for (featureCount <- featureCounts) {
      val count = featureCount._2
      featureCountHistogram.update(count, featureCountHistogram(count) + 1)
    }
    val counts = featureCountHistogram.keys.toSeq.sortBy(key => (featureCountHistogram(key), -key))
    println("Feature histogram:")
    for (count <- counts) {
      println(s"${count} -> ${featureCountHistogram(count)}")
    }
  }

  def selectFeaturesForWords(counts: Counts): Map[Word, Set[Feature]] = {
    counts.allWords.par.map(word => {
      val keptFeatures = selectFeaturesByPmi(word, counts, 100)
      (word, keptFeatures)
    }).seq.toMap
  }

  def selectFeaturesByPmi(word: Word, counts: Counts, toKeep: Int): Set[Feature] = {
    val orderedFeatures = counts.allFeatures.toSeq.sortBy(feature => {
      -counts.getWordFeatureCount(word, feature) / (counts.getWordCount(word) * counts.getFeatureCount(feature))
    })
    println(s"Top 5 for ${word.word}")
    for (f <- orderedFeatures.take(5)) {
      println(s"   ${f.feature}")
    }
    (orderedFeatures.take(toKeep / 2) ++ orderedFeatures.reverse.take(toKeep / 2)).toSet
  }

  def filterFeatures(featuresForMid: Map[Mid, Seq[Feature]], keptFeatures: Set[Feature]): Map[Mid, Seq[Feature]] = {
    featuresForMid.par.mapValues(_.filter(feature => keptFeatures.contains(feature)) ++ Seq(Feature("bias"))).seq.toMap
  }

  def outputFeatureMatrix(featuresForMid: Map[Mid, Seq[Feature]], outfile: String) {
    val writer = fileUtil.getFileWriter(outfile)
    val seen_features = new mutable.HashSet[Feature]
    for (midFeatures <- featuresForMid) {
      val mid = midFeatures._1
      val features = midFeatures._2
      writer.write(mid.mid)
      writer.write("\t")
      for ((feature, i) <- features.zipWithIndex) {
        writer.write(feature.feature)
        seen_features.add(feature)
        if (i < features.size - 1) writer.write(" -#- ")
      }
      writer.write("\n")
    }
    writer.close()
    println(s"Saw ${seen_features.size} features")
  }

  def outputFeatureDictionary(featuresForMid: Map[Mid, Seq[Feature]], dictionaryName: String, outfile: String) {
    val writer = fileUtil.getFileWriter(outfile)
    val features = featuresForMid.flatMap(_._2).toSet
    writer.write(s"(define ${dictionaryName} (list\n")
    for (feature <- features) {
      writer.write("\"")
      writer.write(feature.feature)
      writer.write("\"\n")
    }
    writer.write("))\n")
    writer.close()
  }

  def outputWordFeatures(featuresForWord: Map[Word, Set[Feature]], outfile: String) {
    val writer = fileUtil.getFileWriter(outfile)
    for (entry <- featuresForWord) {
      val word = entry._1.word
      val features = entry._2
      writer.write("${word}\t")
      for ((feature, index) <- features.zipWithIndex) {
        writer.write("${feature.feature}")
        if (index < features.size - 1) {
          writer.write("\t")
        }
      }
      writer.write("\"\n")
    }
    writer.write("))\n")
    writer.close()
  }
}

package org.allenai.semparse.one_off

import edu.cmu.ml.rtw.users.matt.util.FileUtil

import scala.collection.mutable

object do_feature_selection {
  val fileUtil = new FileUtil

  def main(args: Array[String]) {
    println("Selecting MID features")
    featureSelectionFromFile(
      "/Users/mattg/pra/results/semparse/mids/unknown/training_matrix.tsv",
      "/Users/mattg/clone/tacl2015-factorization/data/mid_features.tsv"
    )
    println("Selecting MID pair features")
    featureSelectionFromFile(
      "/Users/mattg/pra/results/semparse/mid_pairs/unknown/training_matrix.tsv",
      "/Users/mattg/clone/tacl2015-factorization/data/mid_pair_features.tsv"
    )
  }

  def featureSelectionFromFile(infile: String, outfile: String) {
    println("Reading features from file")
    val (featuresForKey, featureCounts) = readFeaturesFromFile(infile)
    printTopKeys(featuresForKey, 10)
    println("Selecting features")
    val keptFeatures = selectFeatures(featureCounts)
    println(s"Kept ${keptFeatures.size} features")
    println("Filtering features")
    val filteredFeatures = filterFeatures(featuresForKey, keptFeatures)
    printTopKeys(filteredFeatures, 10)
    println("Outputting feature matrix")
    outputFeatureMatrix(filteredFeatures, outfile)
  }

  def readFeaturesFromFile(infile: String) = {
    val featuresForKey = new mutable.HashMap[String, Seq[String]]
    val featureCounts = new mutable.HashMap[String, Int].withDefaultValue(0)
    for (line <- fileUtil.getLineIterator(infile)) {
      val (keyStr, featuresStr) = line.splitAt(line.indexOf('\t'))
      val key = keyStr.trim
      val features = featuresStr.trim.split(" -#- ").map(_.replace(",1.0", ""))
      featuresForKey(key) = features.toSeq
      for (feature <- features) {
        featureCounts.update(feature, featureCounts(feature) + 1)
      }
    }
    (featuresForKey.toMap, featureCounts.toMap)
  }

  def printTopKeys(featuresForKey: Map[String, Seq[String]], topK: Int) {
    val topKeys = featuresForKey.keys.toSeq.sortBy(key => -featuresForKey(key).size).take(topK)
    println("Top keys:")
    for (key <- topKeys) {
      println(s"${key} -> ${featuresForKey(key).size}")
    }
  }

  def printFeatureCountHistogram(featureCounts: Map[String, Int]) {
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

  def selectFeatures(featureCounts: Map[String, Int]): Set[String] = {
    val features = featureCounts.keySet
    features.par.filter(feature => featureCounts(feature) > 5).toSet.seq
  }

  def filterFeatures(featuresForKey: Map[String, Seq[String]], keptFeatures: Set[String]) = {
    featuresForKey.par.mapValues(_.filter(feature => keptFeatures.contains(feature)) ++ Seq("bias")).seq.toMap
  }

  def outputFeatureMatrix(featuresForKey: Map[String, Seq[String]], outfile: String) {
    val writer = fileUtil.getFileWriter(outfile)
    for (keyFeatures <- featuresForKey) {
      val key = keyFeatures._1
      val features = keyFeatures._2
      writer.write(key)
      writer.write("\t")
      for ((feature, i) <- features.zipWithIndex) {
        writer.write(feature)
        if (i < features.size - 1) writer.write(" -#- ")
      }
      writer.write("\n")
    }
  }
}

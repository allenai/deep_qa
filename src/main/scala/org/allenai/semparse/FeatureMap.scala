package org.allenai.semparse

import edu.cmu.ml.rtw.users.matt.util.FileUtil

class FeatureMap(featureFile: String, fileUtil: FileUtil = new FileUtil) {
  lazy val features: Map[String, Seq[String]] = fileUtil.readMapListFromTsvFile(featureFile)
  def getFeatures(key: String) = features(key)
  def getFeatures(key: String, default: Seq[String]) = features.getOrElse(key, default)
  lazy val allAllowedFeatures = features.values.flatten.toSet
}


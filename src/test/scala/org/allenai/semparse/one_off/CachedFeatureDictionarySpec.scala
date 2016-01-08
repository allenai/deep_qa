package org.allenai.semparse.one_off

import org.scalatest._

import com.jayantkrish.jklol.lisp.CachedFeatureDictionary

class CachedFeatureDictionarySpec extends FlatSpecLike with Matchers {
  "getFeatures" should "return the right number of features" in {
    val featureDict = new CachedFeatureDictionary("data/mid_features.tsv") {
      override def computeFeaturesForKey(key: String): java.util.List[String] = {
        new java.util.ArrayList[String]()
      }
    }
    featureDict.getNumFeatures() should be(20001)
    featureDict.getNumCachedVectors() should be(17062)
  }
}

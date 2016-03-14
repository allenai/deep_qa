package org.allenai.semparse

import org.scalatest._

import com.jayantkrish.jklol.tensor.SparseTensor

import com.mattg.util.FakeFileUtil

class CachedFeatureDictionarySpec extends FlatSpecLike with Matchers {

  val featureFile = "/feature_file.tsv"
  val featureFileContents = """entity1	feature1 -#- feature2 -#- feature3 -#- feature4 -#- bias
    |entity2	feature2 -#- feature3 -#- bias
    |entity3	feature5 -#- feature6 -#- feature7 -#- bias
    |entity4	feature1 -#- feature6 -#- feature8 -#- bias
    |entity5	feature8 -#- bias""".stripMargin
  val wordFeatureDictFile = "/word_feature_dict.tsv"
  val wordFeatureDictFileContents = """word1	feature1	feature3	bias
    |word2	feature2	feature3	bias
    |word3	feature4	feature6	feature7	bias""".stripMargin

  val fileUtil = new FakeFileUtil
  fileUtil.addFileToBeRead(featureFile, featureFileContents)
  fileUtil.addFileToBeRead(wordFeatureDictFile, wordFeatureDictFileContents)
  val dict = new CachedFeatureDictionary(featureFile, wordFeatureDictFile, false, fileUtil) {
    override def computeFeaturesForKey(key: String) = Seq("bias")
  }

  "loadDictionaries" should "load the features for each word" in {
    dict.getNumFeatures("word1") should be(3)
    dict.getNumFeatures("word2") should be(3)
    dict.getNumFeatures("word3") should be(4)
    dict.getNumFeatures("unseen") should be(1)

    dict.isValidFeature("feature1", "word1") should be(true)
    dict.isValidFeature("feature2", "word1") should be(false)
    dict.isValidFeature("feature3", "word1") should be(true)
    dict.isValidFeature("feature4", "word1") should be(false)
    dict.isValidFeature("feature5", "word1") should be(false)
    dict.isValidFeature("feature6", "word1") should be(false)
    dict.isValidFeature("feature7", "word1") should be(false)
    dict.isValidFeature("feature8", "word1") should be(false)
    dict.isValidFeature("bias", "word1") should be(true)
  }

  it should "return correct values with unseen words" in {
    dict.isValidFeature("feature1", "unseen") should be(false)
    dict.isValidFeature("feature2", "unseen") should be(false)
    dict.isValidFeature("feature3", "unseen") should be(false)
    dict.isValidFeature("feature4", "unseen") should be(false)
    dict.isValidFeature("feature5", "unseen") should be(false)
    dict.isValidFeature("feature6", "unseen") should be(false)
    dict.isValidFeature("feature7", "unseen") should be(false)
    dict.isValidFeature("feature8", "unseen") should be(false)
    dict.isValidFeature("bias", "unseen") should be(true)
  }

  "loadFeatures" should "load the features for each entity" in {
    val t1 = dict.getFeatureVector("entity1", "word1").asInstanceOf[SparseTensor]
    t1.getByDimKey(dict.getFeatureIndex("feature1", "word1")) should be(1.0)
    t1.getByDimKey(dict.getFeatureIndex("feature3", "word1")) should be(1.0)
    t1.getByDimKey(dict.getFeatureIndex("bias", "word1")) should be(1.0)

    val t2 = dict.getFeatureVector("entity1", "word2").asInstanceOf[SparseTensor]
    t2.getByDimKey(dict.getFeatureIndex("feature2", "word2")) should be(1.0)
    t2.getByDimKey(dict.getFeatureIndex("feature3", "word2")) should be(1.0)
    t2.getByDimKey(dict.getFeatureIndex("bias", "word2")) should be(1.0)

    val t3 = dict.getFeatureVector("entity1", "word3").asInstanceOf[SparseTensor]
    t3.getByDimKey(dict.getFeatureIndex("feature4", "word3")) should be(1.0)
    t3.getByDimKey(dict.getFeatureIndex("feature6", "word3")) should be(0.0)
    t3.getByDimKey(dict.getFeatureIndex("feature7", "word3")) should be(0.0)
    t3.getByDimKey(dict.getFeatureIndex("bias", "word3")) should be(1.0)
  }

  it should "return correct values for unseen words" in {
    val t = dict.getFeatureVector("entity1", "unseen").asInstanceOf[SparseTensor]
    t.getByDimKey(dict.getFeatureIndex("bias", "unseen")) should be(1.0)
  }

  it should "return correct values for unseen entities" in {
    val t1 = dict.getFeatureVector("unseen", "word1").asInstanceOf[SparseTensor]
    t1.getByDimKey(dict.getFeatureIndex("feature1", "word1")) should be(0.0)
    t1.getByDimKey(dict.getFeatureIndex("feature3", "word1")) should be(0.0)
    t1.getByDimKey(dict.getFeatureIndex("bias", "word1")) should be(1.0)

    val t2 = dict.getFeatureVector("unseen", "unseen").asInstanceOf[SparseTensor]
    t2.getByDimKey(dict.getFeatureIndex("bias", "unseen")) should be(1.0)
  }
}


package org.allenai.semparse.pipeline.acl2016

import org.json4s._
import org.json4s.JsonDSL._

import org.allenai.semparse.pipeline.base._

object Acl2016Experiments {

  // *********************************
  // STEP 1: TRAINING DATA PROCESSING
  // *********************************

  val NEW_LF_TRAINING_DATA_FILE = "data/acl2016-training.txt"
  val NEW_LF_TRAINING_DATA_NAME = "acl2016"
  val OLD_LF_TRAINING_DATA_FILE = "data/tacl2015-training.txt"
  val OLD_LF_TRAINING_DATA_NAME = "tacl2015"

  val newLfTrainingDataParams: JValue =
    ("training data file" -> NEW_LF_TRAINING_DATA_FILE) ~
    ("data name" -> NEW_LF_TRAINING_DATA_NAME) ~
    ("word count threshold" -> 5)

  val newLfTrainingDataSampleParams: JValue =
    ("training data file" -> NEW_LF_TRAINING_DATA_FILE) ~
    ("data name" -> (NEW_LF_TRAINING_DATA_NAME + "_sample")) ~
    ("word count threshold" -> 0) ~
    ("lines to use" -> 1000)

  val oldLfTrainingDataParams: JValue =
    ("training data file" -> OLD_LF_TRAINING_DATA_FILE) ~
    ("data name" -> OLD_LF_TRAINING_DATA_NAME) ~
    ("word count threshold" -> 5)

  val oldLfTrainingDataSampleParams: JValue =
    ("training data file" -> OLD_LF_TRAINING_DATA_FILE) ~
    ("data name" -> (OLD_LF_TRAINING_DATA_NAME + "_sample")) ~
    ("word count threshold" -> 0) ~
    ("lines to use" -> 1000)

  // **********************************
  // STEP 2: FEATURE MATRIX COMPUTATION
  // **********************************

  val SFE_SPEC_FILE = "src/main/resources/sfe_spec.json"

  val newLfTrainingDataFeatureParams: JValue =
    ("training data" -> newLfTrainingDataParams) ~
    ("sfe spec file" -> SFE_SPEC_FILE)

  val newLfTrainingDataSampleFeatureParams: JValue =
    ("training data" -> newLfTrainingDataSampleParams) ~
    ("sfe spec file" -> SFE_SPEC_FILE)

  val oldLfTrainingDataFeatureParams: JValue =
    ("training data" -> oldLfTrainingDataParams) ~
    ("sfe spec file" -> SFE_SPEC_FILE)

  val oldLfTrainingDataSampleFeatureParams: JValue =
    ("training data" -> oldLfTrainingDataSampleParams) ~
    ("sfe spec file" -> SFE_SPEC_FILE)

  // *********************************
  // STEP 3: PMI CALCULATION
  // *********************************

  val newLfTrainingDataPmiParams: JValue =
    ("training data features" -> newLfTrainingDataFeatureParams)
  val newLfTrainingDataSamplePmiParams: JValue =
    ("training data features" -> newLfTrainingDataSampleFeatureParams)
  val oldLfTrainingDataPmiParams: JValue =
    ("training data features" -> oldLfTrainingDataFeatureParams)
  val oldLfTrainingDataSamplePmiParams: JValue =
    ("training data features" -> oldLfTrainingDataSampleFeatureParams)

  // *********************************
  // STEP 4: TRAINING
  // *********************************

  val newLfFormalModelParams: JValue =
    ("model type" -> "formal") ~
    ("feature computer" -> newLfTrainingDataPmiParams)
  val newLfDistributionalModelParams: JValue =
    ("model type" -> "distributional") ~
    ("feature computer" -> newLfTrainingDataPmiParams)
  val newLfCombinedModelParams: JValue =
    ("model type" -> "combined") ~
    ("feature computer" -> newLfTrainingDataPmiParams)

  val oldLfFormalModelParams: JValue =
    ("model type" -> "formal") ~
    ("feature computer" -> oldLfTrainingDataPmiParams)
  val oldLfDistributionalModelParams: JValue =
    ("model type" -> "distributional") ~
    ("feature computer" -> oldLfTrainingDataPmiParams)
  val oldLfCombinedModelParams: JValue =
    ("model type" -> "combined") ~
    ("feature computer" -> oldLfTrainingDataPmiParams)

  val newLfSampleFormalModelParams: JValue =
    ("model type" -> "formal") ~
    ("feature computer" -> newLfTrainingDataSamplePmiParams)
  val newLfSampleDistributionalModelParams: JValue =
    ("model type" -> "distributional") ~
    ("feature computer" -> newLfTrainingDataSamplePmiParams)
  val newLfSampleCombinedModelParams: JValue =
    ("model type" -> "combined") ~
    ("feature computer" -> newLfTrainingDataSamplePmiParams)

  // *********************************
  // STEP 5: TESTING
  // *********************************

  // TODO(matt): Make the test set expansion with Freebase an option here; it currently is just
  // hard-coded to always happen.  So Table 2 doesn't have a nice experiment set up...

  val FINAL_TEST_SET = ("acl2016_test", "src/main/resources/acl2016_final_test_set_annotated.json")
  val DEV_SET = ("acl2016_dev", "src/main/resources/acl2016_dev_set_annotated.json")
  val DEV_SET_OLD_LFS = ("tacl2015_test", "src/main/resources/acl2016_dev_set_old_lfs_annotated.json")

  val newLfFormalModelFinalTestParams: JValue =
    ("test name" -> FINAL_TEST_SET._1) ~
    ("test query file" -> FINAL_TEST_SET._2) ~
    ("model" -> newLfFormalModelParams)
  val newLfDistributionalModelFinalTestParams: JValue =
    ("test name" -> FINAL_TEST_SET._1) ~
    ("test query file" -> FINAL_TEST_SET._2) ~
    ("model" -> newLfDistributionalModelParams)
  val newLfCombinedModelFinalTestParams: JValue =
    ("test name" -> FINAL_TEST_SET._1) ~
    ("test query file" -> FINAL_TEST_SET._2) ~
    ("model" -> newLfCombinedModelParams)

  val newLfFormalModelDevSetParams: JValue =
    ("test name" -> DEV_SET._1) ~
    ("test query file" -> DEV_SET._2) ~
    ("model" -> newLfFormalModelParams)
  val newLfDistributionalModelDevSetParams: JValue =
    ("test name" -> DEV_SET._1) ~
    ("test query file" -> DEV_SET._2) ~
    ("model" -> newLfDistributionalModelParams)
  val newLfCombinedModelDevSetParams: JValue =
    ("test name" -> DEV_SET._1) ~
    ("test query file" -> DEV_SET._2) ~
    ("model" -> newLfCombinedModelParams)

  val oldLfFormalModelTaclTestSetParams: JValue =
    ("test name" -> DEV_SET_OLD_LFS._1) ~
    ("test query file" -> DEV_SET_OLD_LFS._2) ~
    ("model" -> oldLfFormalModelParams)
  val oldLfDistributionalModelTaclTestSetParams: JValue =
    ("test name" -> DEV_SET_OLD_LFS._1) ~
    ("test query file" -> DEV_SET_OLD_LFS._2) ~
    ("model" -> oldLfDistributionalModelParams)
  val oldLfCombinedModelTaclTestSetParams: JValue =
    ("test name" -> DEV_SET_OLD_LFS._1) ~
    ("test query file" -> DEV_SET_OLD_LFS._2) ~
    ("model" -> oldLfCombinedModelParams)

  val newLfSampleFormalModelFinalTestParams: JValue =
    ("test name" -> FINAL_TEST_SET._1) ~
    ("test query file" -> FINAL_TEST_SET._2) ~
    ("model" -> newLfSampleFormalModelParams)
  val newLfSampleDistributionalModelFinalTestParams: JValue =
    ("test name" -> FINAL_TEST_SET._1) ~
    ("test query file" -> FINAL_TEST_SET._2) ~
    ("model" -> newLfSampleDistributionalModelParams)
  val newLfSampleCombinedModelFinalTestParams: JValue =
    ("test name" -> FINAL_TEST_SET._1) ~
    ("test query file" -> FINAL_TEST_SET._2) ~
    ("model" -> newLfSampleCombinedModelParams)


  def main(args: Array[String]) {
    //runFinalTestSetOnSampleData()  // this one's just for testing stuff
    runTacl2015Test()  // Table 1 in the paper
    runDevSet()  // Table 1 and Table 3 in the paper
    runFinalTestSet()  // Table 4 in the paper
  }

  def runTacl2015Test() {
    val methods = Seq(
      ("formal-tacl2015-test", oldLfFormalModelTaclTestSetParams),
      ("distributional-tacl2015-test", oldLfDistributionalModelTaclTestSetParams),
      ("combined-tacl2015-test", oldLfCombinedModelTaclTestSetParams)
    )
    new Evaluator(methods).runPipeline()
  }

  def runDevSet() {
    val methods = Seq(
      ("formal-acl2016-dev", newLfFormalModelDevSetParams),
      ("distributional-acl2016-dev", newLfDistributionalModelDevSetParams),
      ("combined-acl2016-dev", newLfCombinedModelDevSetParams)
    )
    new Evaluator(methods).runPipeline()
  }

  def runFinalTestSet() {
    val methods = Seq(
      ("formal-acl2016", newLfFormalModelFinalTestParams),
      ("distributional-acl2016", newLfDistributionalModelFinalTestParams),
      ("combined-acl2016", newLfCombinedModelFinalTestParams)
    )
    new Evaluator(methods).runPipeline()
  }

  def runFinalTestSetOnSampleData() {
    val methods = Seq(
      ("formal-acl2016-sample", newLfSampleFormalModelFinalTestParams),
      ("distributional-acl2016-sample", newLfSampleDistributionalModelFinalTestParams),
      ("combined-acl2016-sample", newLfSampleCombinedModelFinalTestParams)
    )
    new Evaluator(methods).runPipeline()
  }
}

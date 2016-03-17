package org.allenai.semparse.pipeline

import org.json4s._
import org.json4s.JsonDSL._

object Experiments {

  // *********************************
  // STEP 1: TRAINING DATA PROCESSING
  // *********************************

  val NEW_LF_TRAINING_DATA_FILE = "data/acl2016-training.txt"
  val NEW_LF_TRAINING_DATA_NAME = "acl2016"
  val OLD_LF_TRAINING_DATA_FILE = "data/tacl2015/tacl2015-training.txt"
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

  val FINAL_TEST_SET = "src/main/resources/acl2016_final_test_set_annotated.json"
  val DEV_SET = "src/main/resources/acl2016_dev_set_annotated.json"
  val DEV_SET_OLD_LFS = "src/main/resources/acl2016_dev_set_old_lfs_annotated.json"

  val newLfFormalModelFinalTestParams: JValue =
    ("test query file" -> FINAL_TEST_SET) ~
    ("model" -> newLfFormalModelParams)
  val newLfDistributionalModelFinalTestParams: JValue =
    ("test query file" -> FINAL_TEST_SET) ~
    ("model" -> newLfDistributionalModelParams)
  val newLfCombinedModelFinalTestParams: JValue =
    ("test query file" -> FINAL_TEST_SET) ~
    ("model" -> newLfCombinedModelParams)

  val newLfSampleFormalModelFinalTestParams: JValue =
    ("test query file" -> FINAL_TEST_SET) ~
    ("model" -> newLfFormalModelParams)
  val newLfSampleDistributionalModelFinalTestParams: JValue =
    ("test query file" -> FINAL_TEST_SET) ~
    ("model" -> newLfDistributionalModelParams)
  val newLfSampleCombinedModelFinalTestParams: JValue =
    ("test query file" -> FINAL_TEST_SET) ~
    ("model" -> newLfCombinedModelParams)


  def main(args: Array[String]) {
    runFinalTestSetOnSampleData()
    //runFinalTestSet()
  }

  def runFinalTestSet() {
    val methods = Seq(
      ("formal", newLfFormalModelFinalTestParams),
      ("distributional", newLfDistributionalModelFinalTestParams),
      ("combined", newLfCombinedModelFinalTestParams)
    )
    new Evaluator(methods).runPipeline()
  }

  def runFinalTestSetOnSampleData() {
    val methods = Seq(
      ("formal", newLfSampleFormalModelFinalTestParams),
      ("distributional", newLfSampleDistributionalModelFinalTestParams),
      ("combined", newLfSampleCombinedModelFinalTestParams)
    )
    new Evaluator(methods).runPipeline()
  }
}

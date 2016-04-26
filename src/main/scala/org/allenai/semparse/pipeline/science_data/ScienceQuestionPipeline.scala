package org.allenai.semparse.pipeline.science_data

import org.json4s._
import org.json4s.JsonDSL._

import com.mattg.util.FileUtil

import org.allenai.semparse.pipeline.base._

object ScienceQuestionPipeline {
  val fileUtil = new FileUtil

  //////////////////////////////////////////////////////////
  // Step 1: Taking sentences and generating training data
  //////////////////////////////////////////////////////////

  val sentenceProcessorParams: JValue =
    ("max word count per sentence" -> 10) ~
    ("data name" -> "petert_sentences") ~
    ("data directory" -> "/home/mattg/data/petert_science_sentences")
  val sentenceProcessorType: JValue = ("type" -> "science sentence processor")
  val sentenceProcessorParamsWithType: JValue = sentenceProcessorParams merge sentenceProcessorType

  //////////////////////////////////////////////////////////
  // Step 2: Generate a KB of triples from the parses above
  //////////////////////////////////////////////////////////

  val kbGeneratorParams: JValue = ("sentences" -> sentenceProcessorParams)

  val kbGraphCreatorParams: JValue = ("graph name" -> "petert") ~ ("corpus triples" -> kbGeneratorParams)
  val kbGraphCreatorType: JValue = ("type" -> "kb graph creator")
  val kbGraphCreatorParamsWithType: JValue = kbGraphCreatorParams merge kbGraphCreatorType

  ////////////////////////////////////////////////////////////////
  // Step 3: Processing the training data into Jayant's lisp files
  ////////////////////////////////////////////////////////////////

  val trainingDataParams: JValue =
    ("training data creator" -> sentenceProcessorParamsWithType) ~
    ("data name" -> "science/petert_sentences") ~
    ("lines to use" -> 700000) ~
    ("word count threshold" -> 5)

  ////////////////////////////////////////////////////////////////
  // Step 4: Select features for each word
  ////////////////////////////////////////////////////////////////

  val SFE_SPEC_FILE = "src/main/resources/science_sfe_spec.json"

  val trainingDataFeatureParams: JValue =
    ("training data" -> trainingDataParams) ~
    ("sfe spec file" -> SFE_SPEC_FILE) ~
    ("graph creator" -> kbGraphCreatorParamsWithType)

  val trainingDataPmiParams: JValue =
    ("training data features" -> trainingDataFeatureParams)

  ////////////////////////////////////////////////////////////////
  // Step 5: Train a model
  ////////////////////////////////////////////////////////////////

  val modelParams: JValue =
    ("model type" -> "combined") ~
    ("feature computer" -> trainingDataPmiParams)

  ////////////////////////////////////////////////////////////////
  // Step 6: Process the questions into logical forms
  ////////////////////////////////////////////////////////////////

  val questionProcesserParams: JValue =
    ("question file" -> "data/science/monarch_questions/raw_questions.tsv") ~
    ("data name" -> "monarch_questions")

  /////////////////////////////////////////////////////////////////////
  // Step 7: Score the answer options for each question using the model
  /////////////////////////////////////////////////////////////////////

  val questionScorerParams: JValue =
    ("questions" -> questionProcesserParams) ~
    ("model" -> modelParams)

  def main(args: Array[String]) {
    //new Trainer(modelParams, fileUtil).runPipeline()
    new ScienceQuestionScorer(questionScorerParams, fileUtil).runPipeline()
  }
}

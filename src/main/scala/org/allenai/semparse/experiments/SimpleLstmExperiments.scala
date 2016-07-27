package org.allenai.semparse.experiments

import org.json4s._
import org.json4s.JsonDSL._

import com.mattg.util.FileUtil

import org.allenai.semparse.pipeline.common._
import org.allenai.semparse.pipeline.science_data._

object SimpleLstmExperiments {
  val fileUtil = new FileUtil

  //////////////////////////////////////////////////////////
  // Step 1: Take a corpus and select sentences to use
  //////////////////////////////////////////////////////////

  val sentenceSelectorParams: JValue =
    ("type" -> "sentence selector") ~
    ("sentence selector" -> ("max word count per sentence" -> 10)) ~
    ("max sentences" -> 10000) ~
    ("data name" -> "busc_testing") ~
    ("data directory" -> "s3n://private.store.dev.allenai.org/org.allenai.corpora.busc/extractedDocuments/science_templates")

  //////////////////////////////////////////////////////////////////
  // Step 2: Corrupt the positive sentences to get negative data
  //////////////////////////////////////////////////////////////////

  // Step 2a: train a language model on the positive data.
  val sentenceCorruptorTrainerParams: JValue =
    ("positive data" -> sentenceSelectorParams) ~
    ("tokenize input" -> false) ~
    ("word dimensionality" -> 10) ~
    ("max training epochs" -> 1) ~
    ("maximum training sentences" -> 1000)

  // Step 2b: actually corrupt the data
  val sentenceCorruptorParams: JValue =
    ("type" -> "sentence corruptor") ~
    ("positive data" -> sentenceSelectorParams) ~
    ("trainer" -> sentenceCorruptorTrainerParams)

  ///////////////////////////////////////////////////////////////////////////
  // Step 3: Convert question-answer pairs into sentences for validation data
  ///////////////////////////////////////////////////////////////////////////

  val validationQuestionParams: JValue =
    ("question file" -> "data/science/monarch_questions/raw_questions.tsv") ~
    ("output file" -> "data/science/monarch_questions/processed_questions.txt") ~
    ("wh-movement" -> "matt's")

  ////////////////////////////////////////////////////////////////
  // Step 4: Train a model
  ////////////////////////////////////////////////////////////////

  val modelParams: JValue =
    ("model type" -> "simple lstm") ~
    ("model name" -> "abstracts_testing/simple_lstm") ~
    ("validation questions" -> validationQuestionParams) ~
    ("number of epochs" -> 1) ~
    ("max training instances" -> 1000) ~
    ("positive data" -> sentenceSelectorParams)

  // STUFF BELOW HERE STILL TODO

  /////////////////////////////////////////////////////////////////////
  // Step 7: Score the answer options for each question using the model
  /////////////////////////////////////////////////////////////////////

  val questionScorerParams: JValue =
    ("questions" -> validationQuestionParams) ~
    ("model" -> modelParams)

  def main(args: Array[String]) {
    new SentenceSelector(sentenceSelectorParams, fileUtil).runPipeline()
    //new SentenceToLogic(sentenceToLogicParams, fileUtil).runPipeline()
    //new SentenceCorruptor(sentenceCorruptorParams, fileUtil).runPipeline()
    //new QuestionInterpreter(questionInterpreterParams, fileUtil).runPipeline()
    //NeuralNetworkTrainer.create(modelParams, fileUtil).runPipeline()
  }
}

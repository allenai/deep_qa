package org.allenai.semparse.experiments

import org.json4s._
import org.json4s.JsonDSL._

import com.mattg.util.FileUtil

import org.allenai.semparse.pipeline.common._
import org.allenai.semparse.pipeline.science_data._

object MemoryNetworkExperiments {
  val fileUtil = new FileUtil

  //////////////////////////////////////////////////////////
  // Step 1: Take a corpus and select sentences to use
  //////////////////////////////////////////////////////////

  val sentenceSelectorParams: JValue =
    ("type" -> "sentence selector") ~
    ("create sentence indices" -> true) ~
    ("sentence selector" -> ("max word count per sentence" -> 10)) ~
    ("data name" -> "abstracts_testing") ~
    ("data directory" -> "/home/mattg/data/vu_data/abstracts.txt")

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

  /////////////////////////////////////////////////////////////////////
  // Step 3: Get background passages for the positive and negative data
  /////////////////////////////////////////////////////////////////////

  val baseElasticSearchParams: JValue =
    ("num passages per sentence" -> 10) ~
    ("elastic search index url" -> "aristo-es1.dev.ai2") ~
    ("elastic search index port" -> 9300) ~
    ("elastic search cluster name" -> "aristo-es") ~
    ("elastic search index name" -> "busc")

  val positiveBackgroundParams: JValue = baseElasticSearchParams merge
    (("sentences" -> sentenceSelectorParams): JValue)

  val negativeBackgroundParams: JValue = baseElasticSearchParams merge
    (("sentences" -> sentenceCorruptorParams): JValue)

  // STUFF BELOW HERE STILL TODO

  ////////////////////////////////////////////////////////////////
  // Step 3: Train a model
  ////////////////////////////////////////////////////////////////

  val modelParams: JValue =
    ("model type" -> "combined")

  ////////////////////////////////////////////////////////////////
  // Step 4: Convert question-answer pairs into sentences
  ////////////////////////////////////////////////////////////////

  val questionInterpreterParams: JValue =
    ("question file" -> "data/science/monarch_questions/raw_questions.tsv") ~
    ("output file" -> "data/science/monarch_questions/processed_questions.txt") ~
    ("wh-movement" -> "mark's")

  /////////////////////////////////////////////////////////////////////
  // Step 7: Score the answer options for each question using the model
  /////////////////////////////////////////////////////////////////////

  val questionScorerParams: JValue =
    ("questions" -> questionInterpreterParams) ~
    ("model" -> modelParams)

  def main(args: Array[String]) {
    //new SentenceToLogic(sentenceToLogicParams, fileUtil).runPipeline()
    //new SentenceCorruptor(sentenceCorruptorParams, fileUtil).runPipeline()
    //new QuestionInterpreter(questionInterpreterParams, fileUtil).runPipeline()
    new LuceneBackgroundCorpusSearcher(positiveBackgroundParams, fileUtil).runPipeline()
  }
}

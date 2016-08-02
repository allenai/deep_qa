package org.allenai.semparse.experiments

import org.json4s._
import org.json4s.JsonDSL._

import com.mattg.util.FileUtil

import org.allenai.semparse.pipeline.common._
import org.allenai.semparse.pipeline.science_data._

object MemoryNetworkExperiments {
  val fileUtil = new FileUtil

  // These parameters define where the elastic search index is that we'll get background data from,
  // and how many results to request from the index per query.
  val baseElasticSearchParams: JValue =
    ("num passages per sentence" -> 10) ~
    ("elastic search index url" -> "aristo-es1.dev.ai2") ~
    ("elastic search index port" -> 9300) ~
    ("elastic search cluster name" -> "aristo-es") ~
    ("elastic search index name" -> "science_templates")

  //////////////////////////////////////////////////////////
  // Step 1: Take a corpus and select sentences to use
  //////////////////////////////////////////////////////////

  val corpus = "s3n://private.store.dev.allenai.org/org.allenai.corpora.busc/extractedDocuments/science_templates"
  val sentenceSelectorParams: JValue =
    ("sentence producer type" -> "sentence selector") ~
    ("create sentence indices" -> true) ~
    ("sentence selector" -> ("max word count per sentence" -> 10)) ~
    ("data name" -> "busc_testing") ~
    ("data directory" -> corpus) ~
    ("max sentences" -> 100)

  //////////////////////////////////////////////////////////////////
  // Step 2: Corrupt the positive sentences to get negative data
  //////////////////////////////////////////////////////////////////

  // Step 2a: train a language model on the positive data.
  val languageModelTrainerParams: JValue =
    ("sentences" -> sentenceSelectorParams) ~
    ("tokenize input" -> false) ~
    ("word dimensionality" -> 10) ~
    ("max training epochs" -> 1)

  // Step 2b: actually corrupt the data
  val sentenceCorruptorParams: JValue =
    ("sentence producer type" -> "sentence corruptor") ~
    ("create sentence indices" -> true) ~
    ("positive data" -> sentenceSelectorParams) ~
    ("language model" -> languageModelTrainerParams)

  /////////////////////////////////////////////////////////////////////
  // Step 3: Get background passages for the positive and negative data
  /////////////////////////////////////////////////////////////////////

  val positiveBackgroundParams: JValue = baseElasticSearchParams merge
    (("sentences" -> sentenceSelectorParams): JValue)

  val negativeBackgroundParams: JValue = baseElasticSearchParams merge
    (("sentences" -> sentenceCorruptorParams): JValue)

  ///////////////////////////////////////////////////////////////////////////
  // Step 4: Convert question-answer pairs into sentences for validation data
  ///////////////////////////////////////////////////////////////////////////

  val validationQuestionParams: JValue =
    ("sentence producer type" -> "question interpreter") ~
    ("create sentence indices" -> true) ~
    ("question file" -> "/home/mattg/data/questions/omnibus_train_raw.tsv") ~
    ("output file" -> "data/science/omnibus_questions/processed_questions.tsv") ~
    ("last sentence only" -> false) ~
    ("wh-movement" -> "matt's")

  //////////////////////////////////////////////////////////
  // Step 5: Get background passages for the validation data
  //////////////////////////////////////////////////////////

  val validationBackgroundParams: JValue = baseElasticSearchParams merge
    (("sentences" -> validationQuestionParams): JValue)

  ////////////////////////////////////////////////////////////////
  // Step 6: Train a model
  ////////////////////////////////////////////////////////////////

  val modelParams: JValue =
    ("model type" -> "memory network") ~
    ("model name" -> "abstracts_testing/memory_network") ~
    ("positive data" -> sentenceSelectorParams) ~
    ("positive background" -> positiveBackgroundParams) ~
    ("negative data" -> sentenceCorruptorParams) ~
    ("negative background" -> negativeBackgroundParams) ~
    ("validation questions" -> validationQuestionParams) ~
    ("validation background" -> validationBackgroundParams) ~
    ("number of epochs" -> 1)

  // STUFF BELOW HERE STILL TODO

  /////////////////////////////////////////////////////////////////////
  // Step 7: Score the answer options for each question using the model
  /////////////////////////////////////////////////////////////////////

  val questionScorerParams: JValue =
    ("questions" -> validationQuestionParams) ~
    ("model" -> modelParams)

  def main(args: Array[String]) {
    //new SentenceToLogic(sentenceToLogicParams, fileUtil).runPipeline()
    //new SentenceCorruptor(sentenceCorruptorParams, fileUtil).runPipeline()
    //new QuestionInterpreter(questionInterpreterParams, fileUtil).runPipeline()
    //new LuceneBackgroundCorpusSearcher(positiveBackgroundParams, fileUtil).runPipeline()
    new MemoryNetworkTrainer(modelParams, fileUtil).runPipeline()
  }
}

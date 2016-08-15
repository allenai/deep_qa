package org.allenai.dlfa.experiments

import org.json4s._
import org.json4s.JsonDSL._

import com.mattg.util.FileUtil

import org.allenai.dlfa.pipeline._

object MemoryNetworkExperiments {
  val fileUtil = new FileUtil

  // These parameters define where the elastic search index is that we'll get background data from,
  // and how many results to request from the index per query.
  val baseElasticSearchParams: JValue =
    ("num passages per sentence" -> 4) ~
    ("elastic search index url" -> "aristo-es1.dev.ai2") ~
    ("elastic search index port" -> 9300) ~
    ("elastic search cluster name" -> "aristo-es") ~
    ("elastic search index name" -> "busc")

  //////////////////////////////////////////////////////////
  // Step 1: Take a corpus and select sentences to use
  //////////////////////////////////////////////////////////

  val corpus = "s3n://private.store.dev.allenai.org/org.allenai.corpora.busc/extractedDocuments/science_templates"
  val sentenceSelectorParams: JValue =
    ("sentence producer type" -> "sentence selector") ~
    ("create sentence indices" -> true) ~
    ("sentence selector" -> ("max word count per sentence" -> 8) ~
                            ("min word count per sentence" -> 6)) ~
    ("data name" -> "busc_testing2") ~
    ("data directory" -> corpus) ~
    ("max sentences" -> 100)

  //////////////////////////////////////////////////////////////////
  // Step 2: Corrupt the positive sentences to get negative data
  //////////////////////////////////////////////////////////////////

  // Step 2a: train a language model on the positive data.
  val languageModelParams: JValue =
    ("sentences" -> sentenceSelectorParams) ~
    ("tokenize input" -> false) ~
    ("use lstm" -> true) ~
    ("word dimensionality" -> 10) ~
    ("max training epochs" -> 1)

  // Step 2b: generate candidate corruptions using the KB
  val kbSentenceCorruptorParams: JValue =
    ("positive data" -> sentenceSelectorParams) ~
    ("kb tensor file" -> "/home/mattg/data/aristo_kb/animals-tensor-july10-yesonly-wo-thing.csv")

  // Step 2c: use the language model to select among the candidates
  val corruptedSentenceSelectorParams: JValue =
    ("sentence producer type" -> "kb sentence corruptor") ~
    ("create sentence indices" -> true) ~
    ("candidates per set" -> 1) ~
    ("corruptor" -> kbSentenceCorruptorParams) ~
    ("language model" -> languageModelParams)

  /////////////////////////////////////////////////////////////////////
  // Step 3: Get background passages for the positive and negative data
  /////////////////////////////////////////////////////////////////////

  val positiveBackgroundParams: JValue = baseElasticSearchParams merge
    (("sentences" -> sentenceSelectorParams): JValue)

  val negativeBackgroundParams: JValue = baseElasticSearchParams merge
    ("sentences" -> corruptedSentenceSelectorParams) ~
    ("remove query near duplicates" -> true)

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

  val pretrainedEmbeddingParams: JValue =
    ("file" -> "/home/mattg/data/glove.840B.300d.txt.gz") ~
    ("fine tune" -> true) ~
    ("add projection" -> false)

  val baseModelParams: JValue =
    ("positive data" -> sentenceSelectorParams) ~
    ("negative data" -> corruptedSentenceSelectorParams) ~
    ("validation questions" -> validationQuestionParams) ~
    ("pretrained embeddings" -> pretrainedEmbeddingParams) ~
    ("number of epochs" -> 1)

  val memoryNetworkParams: JValue = baseModelParams merge (
    ("model type" -> "memory network") ~
    ("model name" -> "busc/attentive_memory_network") ~
    ("positive background" -> positiveBackgroundParams) ~
    ("negative background" -> negativeBackgroundParams) ~
    ("validation background" -> validationBackgroundParams)
  )

  val simpleLstmModelParams: JValue = baseModelParams merge (
    ("model type" -> "simple lstm") ~
    ("model name" -> "busc/simple_lstm")
  )

  // STUFF BELOW HERE STILL TODO

  /////////////////////////////////////////////////////////////////////
  // Step 7: Score the answer options for each question using the model
  /////////////////////////////////////////////////////////////////////

  val questionScorerParams: JValue =
    ("questions" -> validationQuestionParams) ~
    ("model" -> memoryNetworkParams)

  def main(args: Array[String]) {
    //new SentenceToLogic(sentenceToLogicParams, fileUtil).runPipeline()
    //new SentenceCorruptor(sentenceCorruptorParams, fileUtil).runPipeline()
    //new QuestionInterpreter(questionInterpreterParams, fileUtil).runPipeline()
    //new LuceneBackgroundCorpusSearcher(positiveBackgroundParams, fileUtil).runPipeline()
    NeuralNetworkTrainer.create(simpleLstmModelParams, fileUtil).runPipeline()
    NeuralNetworkTrainer.create(memoryNetworkParams, fileUtil).runPipeline()
  }
}

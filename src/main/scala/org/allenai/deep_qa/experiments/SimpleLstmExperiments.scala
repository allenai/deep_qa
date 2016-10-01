package org.allenai.deep_qa.experiments

import org.json4s._
import org.json4s.JsonDSL._

import com.mattg.util.FileUtil

import org.allenai.deep_qa.pipeline._

object SimpleLstmExperiments {
  val fileUtil = new FileUtil

  //////////////////////////////////////////////////////////
  // Step 1: Take a corpus and select sentences to use
  //////////////////////////////////////////////////////////

  val sentenceSelectorParams: JValue =
    ("sentence producer type" -> "sentence selector") ~
    ("sentence selector" -> ("max word count per sentence" -> 10)) ~
    ("max sentences" -> 10000) ~
    ("data name" -> "busc_testing") ~
    ("data directory" -> "s3n://private.store.dev.allenai.org/org.allenai.corpora.busc/extractedDocuments/science_templates")

  //////////////////////////////////////////////////////////////////
  // Step 2: Corrupt the positive sentences to get negative data
  //////////////////////////////////////////////////////////////////

  // Step 2a: train a language model on the positive data.
  val languageModelParams: JValue =
    ("sentences" -> sentenceSelectorParams) ~
    ("tokenize input" -> false) ~
    ("word dimensionality" -> 10) ~
    ("max training epochs" -> 1) ~
    ("num sentences to use" -> 1000)

  // Step 2b: generate candidate corruptions using the KB
  val kbSentenceCorruptorParams: JValue =
    ("positive data" -> sentenceSelectorParams) ~
    ("kb tensor file" -> "/home/mattg/data/aristo_kb/animals-tensor-july10-yesonly-wo-thing.csv")

  // Step 2c: use the language model to select among the candidates
  val corruptedSentenceSelectorParams: JValue =
    ("sentence producer type" -> "kb sentence corruptor") ~
    ("corruptor" -> kbSentenceCorruptorParams) ~
    ("language model" -> languageModelParams)

  ///////////////////////////////////////////////////////////////////////////
  // Step 3: Convert question-answer pairs into sentences for validation data
  ///////////////////////////////////////////////////////////////////////////

  val validationQuestionParams: JValue =
    ("sentence producer type" -> "question interpreter") ~
    ("question file" -> "/home/mattg/data/questions/omnibus_train_raw.tsv") ~
    ("output file" -> "data/science/omnibus_questions/processed_questions.tsv") ~
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
    ("positive data" -> sentenceSelectorParams) ~
    ("negative data" -> corruptedSentenceSelectorParams)

  // STUFF BELOW HERE STILL TODO

  /////////////////////////////////////////////////////////////////////
  // Step 7: Score the answer options for each question using the model
  /////////////////////////////////////////////////////////////////////

  val questionScorerParams: JValue =
    ("questions" -> validationQuestionParams) ~
    ("model" -> modelParams)

  def main(args: Array[String]) {
    //new SentenceSelector(sentenceSelectorParams, fileUtil).runPipeline()
    //new SentenceToLogic(sentenceToLogicParams, fileUtil).runPipeline()
    //new CorruptedSentenceSelector(corruptedSentenceSelectorParams, fileUtil).runPipeline()
    //new QuestionInterpreter(questionInterpreterParams, fileUtil).runPipeline()
    NeuralNetworkTrainer.create(modelParams, fileUtil).runPipeline()
  }
}

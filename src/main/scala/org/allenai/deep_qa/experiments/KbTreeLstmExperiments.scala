package org.allenai.deep_qa.experiments

import org.json4s._
import org.json4s.JsonDSL._

import com.mattg.util.FileUtil

import org.allenai.deep_qa.pipeline._

object KbTreeLstmExperiments {
  val fileUtil = new FileUtil

  //////////////////////////////////////////////////////////
  // Step 1: Take a corpus and select sentences to use
  //////////////////////////////////////////////////////////

  val sentenceSelectorParams: JValue =
    ("sentence selector" -> ("max word count per sentence" -> 10)) ~
    ("data name" -> "abstracts_testing") ~
    ("data directory" -> "/home/mattg/data/vu_data/abstracts.txt")

  //////////////////////////////////////////////////////////////////
  // Step 2: Convert sentences to logical forms
  //////////////////////////////////////////////////////////////////

  val sentenceToLogicParams: JValue =
    ("sentences" -> sentenceSelectorParams) ~
    ("logical forms" -> ("nested" -> true))

  //////////////////////////////////////////////////////////////////
  // Step 3: Corrupt the positive logical forms to get negative data
  //////////////////////////////////////////////////////////////////

  val logicCorruptorParams: JValue =
    ("positive data" -> sentenceToLogicParams)

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
    new LogicCorruptor(logicCorruptorParams, fileUtil).runPipeline()
  }
}

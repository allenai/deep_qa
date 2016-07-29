package org.allenai.semparse.pipeline.science_data

import org.json4s._

import com.mattg.pipeline.Step
import com.mattg.pipeline.SubprocessStep
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

/**
 * This Step takes as input a collection of sentences produced by a SentenceProducer and trains a
 * language model on those sentences.  The output is a saved model that can be used for whatever
 * you want.
 */
class LanguageModelTrainer(
  params: JValue,
  fileUtil: FileUtil
) extends SubprocessStep(Some(params), fileUtil) {
  implicit val formats = DefaultFormats
  override val name = "Language Model Trainer"

  val validParams = Seq("positive data", "maximum training sentences", "word dimensionality",
    "factor base", "tokenize input", "use lstm", "max training epochs")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val maxSentences = JsonHelper.extractAsOption[Int](params, "maximum training sentences")
  val maxSentencesArgs = maxSentences.map(max => Seq("--max_instances", max.toString)).toSeq.flatten
  val wordDimensionality = JsonHelper.extractWithDefault(params, "word dimensionality", 50)
  val factorBase = JsonHelper.extractWithDefault(params, "factor base", 2)
  val tokenizeInput = JsonHelper.extractWithDefault(params, "tokenize input", true)
  val tokenizeInputArg = if (tokenizeInput) Seq() else Seq("--no_tokenize")
  val useLstm = JsonHelper.extractWithDefault(params, "use lstm", false)
  val useLstmArg = if (useLstm) Seq("--use_lstm") else Seq()
  val trainingEpochs = JsonHelper.extractWithDefault(params, "max training epochs", 20)

  val sentenceProducer = SentenceProducer.create(params \ "positive data", fileUtil)
  val positiveDataFile = sentenceProducer.outputFile
  val modelPrefix = positiveDataFile.dropRight(4) + "_corruption_model"

  override val binary = "python"

  // TODO(matt): might want to change the name of this file to something more general, if we're
  // using this to train general language models.
  override val scriptFile = Some("src/main/python/sentence_corruption/lexical_substitution.py")
  override val arguments = Seq(
    "--train_file", positiveDataFile,
    "--word_dim", wordDimensionality.toString,
    "--factor_base", factorBase.toString,
    "--num_epochs", trainingEpochs.toString,
    "--model_serialization_prefix", modelPrefix
  ) ++ tokenizeInputArg ++ useLstmArg ++ maxSentencesArgs

  override val inputs: Set[(String, Option[Step])] = Set((positiveDataFile, Some(sentenceProducer)))
  override val outputs = Set(
    modelPrefix + "_di.pkl",
    modelPrefix + "_config.json",
    modelPrefix + "_weights.h5"
  )
  override val paramFile = modelPrefix + "_params.json"
  override val inProgressFile = modelPrefix + "_in_progress"
}

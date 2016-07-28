package org.allenai.semparse.pipeline.science_data

import org.json4s._

import com.mattg.pipeline.Step
import com.mattg.pipeline.SubprocessStep
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

/**
 * The goal of this class is to take some positive training data as input, then produce a set of
 * corrupted data for each positive instance, so we can train a model using noise contrastive
 * estimation.
 *
 * INPUTS: a file containing good sentences, one sentence per line.  The input can optionally have
 * indices for each sentence.  The expected file format is either "[sentence]" or
 * "[index][tab][sentence]".
 *
 * OUTPUTS: a file containing corrupted sentences, one sentence per line.  Output format is the
 * same as input format: either "[sentence]" or "[index][tab][sentence]".  Note that the indices
 * are not guaranteed to match between input good statement and output bad statement.  If we
 * actually need that at some point, I'll add it.
 */
class SentenceCorruptor(
  val params: JValue,
  val fileUtil: FileUtil
) extends SubprocessStep(Some(params), fileUtil) with SentenceProducer {
  implicit val formats = DefaultFormats
  override val name = "Sentence Corruptor"

  val validParams = baseParams ++ Seq("positive data", "trainer")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val trainer = new SentenceCorruptorTrainer(params \ "trainer", fileUtil)
  val factorBase = JsonHelper.extractWithDefault(params, "factor base", 2)
  val tokenizeInput = JsonHelper.extractWithDefault(params, "tokenize input", true)
  val tokenizeInputArg = if (tokenizeInput) Seq() else Seq("--no_tokenize")
  val useLstm = JsonHelper.extractWithDefault(params, "use lstm", false)
  val useLstmArg = if (useLstm) Seq("--use_lstm") else Seq()
  val trainingEpochs = JsonHelper.extractWithDefault(params, "max training epochs", 20)
  val maxSentencesArgs = maxSentences.map(max => Seq("--max_corrupted_instances", max.toString)).toSeq.flatten
  val indexSentencesArg = if (indexSentences) Seq("--create_sentence_indices") else Seq()

  val sentenceProducer = SentenceProducer.create(params \ "positive data", fileUtil)
  val positiveDataFile = sentenceProducer.outputFile
  override val outputFile = positiveDataFile.dropRight(4) + "_corrupted.tsv"

  override val binary = "python"
  override val scriptFile = Some("src/main/python/sentence_corruption/lexical_substitution.py")
  override val arguments = Seq(
    "--test_file", positiveDataFile,
    "--output_file", outputFile,
    "--word_dim", trainer.wordDimensionality.toString,
    "--factor_base", trainer.factorBase.toString,
    "--model_serialization_prefix", trainer.modelPrefix
  ) ++ tokenizeInputArg ++ useLstmArg ++ maxSentencesArgs ++ indexSentencesArg

  override val inputs: Set[(String, Option[Step])] = Set(
    (positiveDataFile, Some(sentenceProducer))
  ) ++ trainer.outputs.map(file => (file, Some(trainer)))
  override val outputs = Set(outputFile)
  override val paramFile = outputFile.dropRight(4) + "_params.json"
  override val inProgressFile = outputFile.dropRight(4) + "_in_progress"
}

class SentenceCorruptorTrainer(
  params: JValue,
  fileUtil: FileUtil
) extends SubprocessStep(Some(params), fileUtil) {
  implicit val formats = DefaultFormats
  override val name = "Sentence Corruptor Trainer"

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

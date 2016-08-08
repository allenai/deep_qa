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

  val validParams = baseParams ++ Seq("positive data", "language model")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val trainer = new LanguageModelTrainer(params \ "language model", fileUtil)
  val tokenizeInputArg = if (trainer.tokenizeInput) Seq() else Seq("--no_tokenize")
  val useLstmArg = if (trainer.useLstm) Seq("--use_lstm") else Seq()

  // These two arguments are defined and extracted in SentenceProducer.
  val maxSentencesArgs = maxSentences.map(max => Seq("--max_corrupted_instances", max.toString)).toSeq.flatten
  val indexSentencesArg = if (indexSentences) Seq("--create_sentence_indices") else Seq()

  val positiveDataProducer = SentenceProducer.create(params \ "positive data", fileUtil)
  val positiveDataFile = positiveDataProducer.outputFile
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
    (positiveDataFile, Some(positiveDataProducer))
  ) ++ trainer.outputs.map(file => (file, Some(trainer)))
  override val outputs = Set(outputFile)
  override val paramFile = outputFile.dropRight(4) + "_params.json"
  override val inProgressFile = outputFile.dropRight(4) + "_in_progress"
}

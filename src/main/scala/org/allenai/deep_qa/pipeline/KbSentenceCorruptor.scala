package org.allenai.dlfa.pipeline

import org.json4s._

import com.mattg.pipeline.Step
import com.mattg.pipeline.SubprocessStep
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

/**
 * The goal of this series of Steps is to take some positive training data as input, then produce a
 * set of corrupted data for each positive instance, so we can train a model using noise
 * contrastive estimation.
 *
 * The way that we do this is a three step process:
 *   1. Given a bunch of positive sentences, we use a KB-backed method to generate similar
 *      sentences that we believe to be false.
 *   2. We train a language model on the positive sentences.
 *   3. We use the language model to select the most plausible of the false sentences, and return
 *      that.
 *
 * The input/output spec for these three steps, then, matches that of the SentenceCorruptor:
 *
 * INPUTS: a file containing good sentences, one sentence per line.  The input can optionally have
 * indices for each sentence.  The expected file format is either "[sentence]" or
 * "[index][tab][sentence]".
 *
 * OUTPUTS: a file containing corrupted sentences, one sentence per line.  Output format is the
 * same as input format: either "[sentence]" or "[index][tab][sentence]".  Note that the indices
 * are not guaranteed to match between input good statement and output bad statement.  If we
 * actually need that at some point, I'll add it.
 *
 * These inputs and outputs are actually spread across the three steps, though.
 * KbSentenceCorruptor takes the input, as does the LanguageModelTrainer.
 * CorruptedSentenceSelector then takes the outputs from those two steps and gives the output
 * listed above; CorruptedSentenceSelector is the SentenceProducer Step that actually gets
 * called by other parts of the pipeline code.
 */
class CorruptedSentenceSelector(
  val params: JValue,
  val fileUtil: FileUtil
) extends SubprocessStep(Some(params), fileUtil) with SentenceProducer {
  implicit val formats = DefaultFormats
  override val name = "Corrupted Sentence Selector"

  val validParams = baseParams ++ Seq("language model", "corruptor", "candidates per set")
  JsonHelper.ensureNoExtras(params, name, validParams)

  // The input file is tab-separated, with multiple candidates per line.  We keep this many of the
  // candidates per line.
  val candidatesToKeepPerSet = JsonHelper.extractWithDefault(params, "candidates per set", 1)

  val trainer = new LanguageModelTrainer(params \ "language model", fileUtil)
  val tokenizeInputArg = if (trainer.tokenizeInput) Seq() else Seq("--no_tokenize")

  // These two arguments are defined and extracted in SentenceProducer.
  val maxSentencesArgs = maxSentences.map(max => Seq("--max_output_sentences", max.toString)).toSeq.flatten
  val indexSentencesArg = if (indexSentences) Seq("--create_sentence_indices") else Seq()

  val kbSentenceCorruptor = new KbSentenceCorruptor(params \ "corruptor", fileUtil)
  val possibleCorruptions = kbSentenceCorruptor.outputFile
  override val outputFile = kbSentenceCorruptor.positiveDataFile.dropRight(4) + "_corrupted.tsv"

  override val binary = "python"
  override val scriptFile = Some("src/main/python/dlfa/sentence_corruption/language_model.py")
  override val arguments = Seq(
    "--candidates_file", possibleCorruptions,
    "--output_file", outputFile,
    "--keep_top_k", candidatesToKeepPerSet.toString,
    "--factor_base", trainer.factorBase.toString,
    "--model_serialization_prefix", trainer.modelPrefix
  ) ++ tokenizeInputArg ++ maxSentencesArgs ++ indexSentencesArg

  override val inputs: Set[(String, Option[Step])] = Set(
    (possibleCorruptions, Some(kbSentenceCorruptor))
  ) ++ trainer.outputs.map(file => (file, Some(trainer)))
  override val outputs = Set(outputFile)
  override val paramFile = outputFile.dropRight(4) + "_params.json"
  override val inProgressFile = outputFile.dropRight(4) + "_in_progress"
}

/**
 * This Step takes sentences and uses a KB to generate corruptions that are plausible but false.
 * See comments on CorruptedSentenceSelector (and the comments in the python code this calls) for a
 * little more detail.
 *
 * INPUTS: (1) positive sentences generated from some SentenceProducer (possibly with sentence
 * indices), (2) a KB file (TODO(matt): document the expected KB file format)
 *
 * OUTPUTS: a file containing possible corruptions of the positive sentences.  For each sentence,
 * we group all possible corruptions onto the same line, separated by tabs.  This is so that the
 * CorruptedSentenceSelector can pick one (or several) of the alternatives from each line using a
 * language model.
 */
class KbSentenceCorruptor(
  val params: JValue,
  val fileUtil: FileUtil
) extends SubprocessStep(Some(params), fileUtil) {
  implicit val formats = DefaultFormats
  override val name = "KB Sentence Corruptor"

  val validParams = Seq("positive data", "kb tensor file", "num corruptions")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val kbTensorFile = (params \ "kb tensor file").extract[String]
  val numCorruptions = JsonHelper.extractWithDefault(params, "num corruptions", 10)

  val sentenceProducer = SentenceProducer.create(params \ "positive data", fileUtil)
  val positiveDataFile = sentenceProducer.outputFile
  val outputFile = positiveDataFile.dropRight(4) + "_potential_corruptions.tsv"

  override val binary = "python"
  override val scriptFile = Some("src/main/python/dlfa/sentence_corruption/kb_sentence_corruptor.py")
  override val arguments = Seq(
    "--input_file", positiveDataFile,
    "--output_file", outputFile,
    "--kb_tensor_file", kbTensorFile,
    "--num_perturbation", numCorruptions.toString
  )

  override val inputs: Set[(String, Option[Step])] = Set(
    (positiveDataFile, Some(sentenceProducer)),
    (kbTensorFile, None)
  )
  override val outputs = Set(outputFile)
  override val paramFile = outputFile.dropRight(4) + "_params.json"
  override val inProgressFile = outputFile.dropRight(4) + "_in_progress"
}

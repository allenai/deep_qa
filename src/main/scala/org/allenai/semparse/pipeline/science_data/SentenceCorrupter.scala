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
  params: JValue,
  fileUtil: FileUtil
) extends SubprocessStep(Some(params), fileUtil) {
  implicit val formats = DefaultFormats
  override val name = "Sentence Corruptor"

  val validParams = Seq("positive data", "create sentence indices")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val indexSentences = JsonHelper.extractWithDefault(params, "create sentence indices", false)

  val sentenceSelector = new SentenceSelectorStep(params \ "positive data", fileUtil)
  val positiveDataFile = sentenceSelector.outputFile
  val outputFile = positiveDataFile.dropRight(4) + "_corrupted.tsv"

  override val binary = "python"
  override val scriptFile = ""  // TODO(matt)
  override val arguments = Seq(
    // TODO(matt)
  )

  override val inputs: Set[(String, Option[Step])] = Set((positiveDataFile, Some(sentenceSelector)))
  override val outputs = Set(outputFile)
  override val paramFile = outputs.head.dropRight(4) + "_params.json"
  override val inProgressFile = outputs.head.dropRight(4) + "_in_progress"
}

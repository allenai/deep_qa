package org.allenai.semparse.pipeline.science_data

import org.json4s._

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

/**
 * This step in the pipeline takes as input a training data file, and produces as output a relation
 * file, containing triples for all entities and relations seen enough times in the training data.
 */
class KbGenerator(
  params: JValue,
  fileUtil: FileUtil = new FileUtil
) extends Step(Some(params), fileUtil) {
  override val name = "KB Generator"

  val validParams = Seq("min np count", "sentences", "output file")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val sentenceProcessor = new ScienceSentenceProcessor(params \ "sentences")
  val trainingDataFile = sentenceProcessor.outputFile
  val tripleFile = JsonHelper.extractWithDefault(params, "output file", "data/science_triples.tsv")

  override val inputs: Set[(String, Option[Step])] = Set(
    (trainingDataFile, Some(sentenceProcessor))
  )
  override val outputs = Set(tripleFile)
  override val paramFile = tripleFile.replace(".tsv", "_params.json")

  override def _runStep() {
  }
}

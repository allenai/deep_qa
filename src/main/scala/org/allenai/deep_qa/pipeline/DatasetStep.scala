package org.allenai.deep_qa.pipeline

import org.json4s._

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

/**
 * This Step collects a number of SentenceProducers into a set of files needed to train a model.
 * There is no work done here; this Step exists simply so that a NeuralNetworkTrainer has a
 * consistent interface to data files, no matter how many of them there are or where they come
 * from.  We just pass all of the work on to whatever SentenceProducers are specified in `params`.
 */
class DatasetStep(
  params: JValue,
  fileUtil: FileUtil
) extends Step(None, fileUtil) {
  implicit val formats = DefaultFormats
  override val name = "DatasetStep"
  val validParams = Seq("data files")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val datasetHash = params.hashCode.toHexString
  val sentenceProducers = (params \ "data files").extract[List[JValue]].map(producerParams => {
    SentenceProducer.create(producerParams, fileUtil)
  })

  override val inputs: Set[(String, Option[Step])] = sentenceProducers.flatMap(producer => {
    producer.outputs.map((_, Some(producer)))
  }).toSet
println(s"Dataset inputs: $inputs")
  val files = sentenceProducers.flatMap(_.outputs.toSeq)
  override val outputs = files.toSet
  override val inProgressFile = outputs.head.dropRight(4) + s"_${datasetHash}_in_progress"

  override def _runStep() { }
}

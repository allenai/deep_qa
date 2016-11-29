package org.allenai.deep_qa.pipeline

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper
import org.json4s._

import scala.collection.mutable

/**
 * A DatasetStep creates the files associated with a dataset.  This Step itself doesn't really do
 * much work; probably there are some SentenceProducers underlying this Step that actually do the
 * work.  This just groups files together so that downstream Steps have a consistent interface to
 * data files, no matter how many of them there are or where they came from.
 *
 * If the concrete Step class does any actual work, we'll want to save parameters for it (so that,
 * e.g., we know exactly which datasets were combined by a DatasetCombinerStep).  If the dataset is
 * just a shell on top of SentenceProducers, we don't need to save parameters.  So there's a
 * `stepHasItsOwnParams` argument here that controls whether we will save parameters for this Step
 * or not.
 */
abstract class DatasetStep(
  params: JValue,
  fileUtil: FileUtil,
  stepHasItsOwnParams: Boolean
) extends Step(if (stepHasItsOwnParams) Some(params) else None, fileUtil) {
  implicit val formats = DefaultFormats
  override val name = "DatasetStep"
  val baseParams = Seq("dataset type")
  val datasetHash = params.hashCode.toHexString

  // The DatasetCombinerStep needs to be able to access a consistent ordering of the output files,
  // so it can combine them correctly.
  def outputFiles: Seq[String]
  override lazy val outputs = outputFiles.toSet
  override lazy val inProgressFile = outputs.head.dropRight(4) + s"_${datasetHash}_in_progress"

  override def _runStep() { }
}

object DatasetStep {
  def create(params: JValue, fileUtil: FileUtil): DatasetStep = {
    val stepType = JsonHelper.extractWithDefault(params, "dataset type", "from sentence producers")
    stepType match {
      case "from sentence producers" => new DatasetFromSentenceProducersStep(params, fileUtil)
      case "combined" => new DatasetCombinerStep(params, fileUtil)
      case "truncated" => new DatasetTruncatorStep(params, fileUtil)
      case _ => throw new IllegalStateException(s"Unrecognized DatasetStep type: $stepType")
    }
  }
}

/**
 * This Step collects a number of SentenceProducers into a set of files needed to train a model.
 * There is no work done here; we just pass all of the work on to whatever SentenceProducers are
 * specified in `params`.
 */
class DatasetFromSentenceProducersStep(
  params: JValue,
  fileUtil: FileUtil
) extends DatasetStep(params, fileUtil, false) {
  val validParams = baseParams ++ Seq("data files")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val sentenceProducers = (params \ "data files").extract[List[JValue]].map(producerParams => {
    SentenceProducer.create(producerParams, fileUtil)
  })

  override val inputs: Set[(String, Option[Step])] = sentenceProducers.flatMap(producer => {
    producer.outputs.map((_, Some(producer)))
  }).toSet
  override val outputFiles = sentenceProducers.flatMap(_.outputs.toSeq)
}

/**
 * This Step truncates a Dataset to only include a fixed number of instances.  Our python solver
 * code has the ability to do this truncation also, so this Step is unnecessary in a lot of cases.
 * One place where it is useful is when used in conjunction with a DatasetCombinerStep - say you
 * have two datasets, and you want to merge part of one with all of the other.  This would be hard
 * in the python code, but not too bad with combinations of DatasetSteps here.
 */
class DatasetTruncatorStep(
  params: JValue,
  fileUtil: FileUtil
) extends DatasetStep(params, fileUtil, true) {
  val validParams = baseParams ++ Seq("dataset to truncate", "output directory", "instances to keep")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val dataset = DatasetStep.create(params \ "dataset to truncate", fileUtil)
  val outputDirectory = (params \ "output directory").extract[String]
  val numInstancesToKeep = (params \ "instances to keep").extract[Int]

  override val paramFile = outputDirectory + "params.json"
  override val inputs: Set[(String, Option[Step])] = dataset.outputs.map((_, Some(dataset)))
  override val outputFiles = dataset.outputFiles.map(outputDirectory + _.split("/").last)

  override def _runStep() {
    val lines: Seq[Seq[(Option[Int], String)]] =
      dataset.outputFiles.map(file => fileUtil.readLinesFromFile(file).map(line => {
        val fields = line.split("\t")
        val (index, strippedLine) = if (fields(0).forall(_.isDigit)) {
          (Some(fields(0).toInt), line)
        } else {
          (None, line)
        }
        (index, strippedLine)
      }))
    if (lines.flatten.exists(_._1.nonEmpty)) {
      // These instances are indexed - we'll use the first file to construct a set of ids to keep,
      // then use that to filter all of the files.
      if (lines.flatten.exists(_._1.isEmpty)) {
        throw new IllegalStateException("Only some of your files have indices; not sure what to do here")
      }
      val idsToKeep = new mutable.HashSet[Int]
      val ids = lines.head.map(_._1.get)
      var index = 0
      while (idsToKeep.size < numInstancesToKeep && index < ids.length) {
        idsToKeep += ids(index)
        index += 1
      }
      for ((fileLines, filename) <- lines.zip(outputFiles)) {
      val outputLines = fileLines.filter(line => idsToKeep.contains(line._1.get)).map(_._2)
        fileUtil.writeLinesToFile(filename, outputLines)
      }
    } else {
      // These instances are not indexed - we'll just take the first N lines from each file.
      for ((fileLines, filename) <- lines.zip(outputFiles)) {
        fileUtil.writeLinesToFile(filename, fileLines.map(_._2).take(numInstancesToKeep))
      }
    }
  }
}

/**
 * This Step combines the output of two or more Dataset steps into a single set of data files.
 * There is some slightly complicated logic to make sure we're concatenating the right files
 * together, and to handle the instance indices in the files, if they exist.
 */
class DatasetCombinerStep(
  params: JValue,
  fileUtil: FileUtil
) extends DatasetStep(params, fileUtil, true) {
  val validParams = baseParams ++ Seq("datasets", "output directory")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val datasets = (params \ "datasets").extract[List[JValue]].map(datasetParams => {
    DatasetStep.create(datasetParams, fileUtil)
  })
  val outputDirectory = (params \ "output directory").extract[String]

  override val paramFile = outputDirectory + "params.json"
  override val inputs: Set[(String, Option[Step])] = datasets.flatMap(dataset => {
    dataset.outputs.map((_, Some(dataset)))
  }).toSet
  override val outputFiles = datasets.head.outputFiles.map(outputDirectory + _.split("/").last)

  override def _runStep() {
    val lines: Seq[Seq[Seq[(Int, Option[Int], String)]]] = (datasets.zipWithIndex.par.map { case (dataset, datasetIndex) => {
      dataset.outputFiles.map(file => fileUtil.readLinesFromFile(file).map(line => {
        val fields = line.split("\t")
        val (index, strippedLine) = if (fields(0).forall(_.isDigit)) {
          (Some(fields(0).toInt), fields.drop(1).mkString("\t"))
        } else {
          (None, line)
        }
        (datasetIndex, index, strippedLine)
      }))
    }}).seq
    val indexMapping: Option[Map[(Int, Int), Int]] = {
      val instanceIndices = (lines.flatten.flatten.map { case (datasetIndex, instanceIndex, line) => (datasetIndex, instanceIndex) }).toSet
      if (instanceIndices.exists(_._2.nonEmpty)) {
        val mapping = new mutable.HashMap[(Int, Int), Int]
        if (instanceIndices.exists(_._2.isEmpty)) {
          throw new IllegalStateException("Only some of your files have indices; cannot combine them")
        }
        var currentIndex = 0
        for (instanceIndex <- instanceIndices) {
          mapping((instanceIndex._1, instanceIndex._2.get)) = currentIndex
          currentIndex += 1
        }
        Some(mapping.toMap)
      } else {
        None
      }
    }
    for ((fileLines, filename) <- lines.transpose.map(_.flatten).zip(outputFiles)) {
      val outputLines = indexMapping match {
        case Some(mapping) => {
          fileLines.map { case (datasetIndex, instanceIndexOption, line) => {
            val newIndex = mapping((datasetIndex, instanceIndexOption.get))
            s"${newIndex}\t$line"
          }}
        }
        case None => {
          fileLines.map(_._3)
        }
      }
      fileUtil.writeLinesToFile(filename, outputLines)
    }
  }
}

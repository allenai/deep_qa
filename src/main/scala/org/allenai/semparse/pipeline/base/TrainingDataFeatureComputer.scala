package org.allenai.semparse.pipeline.base

import edu.cmu.ml.rtw.pra.experiments.Outputter
import edu.cmu.ml.rtw.pra.experiments.RelationMetadata
import edu.cmu.ml.rtw.pra.data.NodeInstance
import edu.cmu.ml.rtw.pra.data.NodePairInstance
import edu.cmu.ml.rtw.pra.features.NodePairSubgraphFeatureGenerator
import edu.cmu.ml.rtw.pra.features.NodeSubgraphFeatureGenerator
import edu.cmu.ml.rtw.pra.graphs.Graph
import edu.cmu.ml.rtw.pra.graphs.Graph

import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper
import com.mattg.util.SpecFileReader
import com.mattg.pipeline.Step

import org.json4s._

import scala.collection.mutable

// This class is very similar to SfeFeatureComputer, but here instead of computing (filtered)
// features and feature vectors to be used during the training and testing of the semantic parser,
// we are computing full feature vectors that will be used as input to the PMI pipeline that
// selects features for each word.  So we don't need nearly the complexity that is in
// SfeFeatureComputer.
class PreFilteredFeatureComputer(
  specFile: String,
  fileUtil: FileUtil = new FileUtil
) {
  implicit val formats = DefaultFormats
  val params = new SpecFileReader("/dev/null").readSpecFile(specFile)

  // Here we need to set up some stuff to use the SFE code.  The first few things don't matter, but
  // we'll use the graph and feature generators in the code below.
  val praBase = "/dev/null"
  val relation = "relation doesn't matter"
  val outputter = Outputter.justLogger
  val relationMetadata = new RelationMetadata(JNothing, praBase, outputter, fileUtil)
  val graph = Graph.create(params \ "graph", praBase, outputter, fileUtil).get
  val nodeFeatureGenerator = new NodeSubgraphFeatureGenerator(
    params \ "node features",
    relation,
    relationMetadata,
    outputter,
    fileUtil
  )
  val nodePairFeatureGenerator = new NodePairSubgraphFeatureGenerator(
    params \ "node pair features",
    relation,
    relationMetadata,
    outputter,
    fileUtil
  )

  def computeMidFeatures(mid: String): Seq[String] = {
    if (graph.hasNode(mid)) {
      val instance = new NodeInstance(graph.getNodeIndex(mid), true, graph)
      val subgraph = nodeFeatureGenerator.getLocalSubgraph(instance)
      nodeFeatureGenerator.extractFeaturesAsStrings(instance, subgraph)
    } else {
      Seq()
    }
  }

  def computeMidPairFeatures(mid1: String, mid2: String): Seq[String] = {
    if (graph.hasNode(mid1) && graph.hasNode(mid2)) {
      val instance = new NodePairInstance(graph.getNodeIndex(mid1), graph.getNodeIndex(mid2), true, graph)
      val subgraph = nodePairFeatureGenerator.getLocalSubgraph(instance)
      nodePairFeatureGenerator.extractFeaturesAsStrings(instance, subgraph)
    } else {
      Seq()
    }
  }
}

class TrainingDataFeatureComputer(
  params: JValue,
  fileUtil: FileUtil = new FileUtil
) extends Step(Some(params), fileUtil) {
  implicit val formats = DefaultFormats

  val validParams = Seq("training data", "sfe spec file")
  JsonHelper.ensureNoExtras(params, "feature matrix computer", validParams)

  val trainingDataParams = (params \ "training data")
  val sfeSpecFile = (params \ "sfe spec file").extract[String]
  val trainingDataProcessor = new TrainingDataProcessor(trainingDataParams, fileUtil)
  val dataName = trainingDataProcessor.dataName
  val outDir = s"data/$dataName"

  val midFile = s"${outDir}/training_mids.tsv"
  val midPairFile = s"${outDir}/training_mid_pairs.tsv"
  val midFeatureFile = s"${outDir}/pre_filtered_mid_features.tsv"
  val midPairFeatureFile = s"${outDir}/pre_filtered_mid_pair_features.tsv"

  override def paramFile = s"$outDir/tdfc_params.json"
  override def name = "Training data feature computer"
  override def inputs = Set(
    (midFile, Some(trainingDataProcessor)),
    (midPairFile, Some(trainingDataProcessor))
  )
  override def outputs = Set(midFeatureFile, midPairFeatureFile)

  override def _runStep() {
    processInMemory()
  }

  def processInMemory() {
    val fileUtil = new FileUtil
    val computer = new PreFilteredFeatureComputer(sfeSpecFile, fileUtil)

    {
      // This block is so that midFeatures goes out of scope and can be garbage collected once it's
      // written to disk.
      println("Computing MID features")
      val midFeatures = fileUtil.parMapLinesFromFile(midFile, (line: String) => {
        val mid = line
        val features = computer.computeMidFeatures(mid)
        val featureStr = features.mkString(" -#- ")
        s"${mid}\t${featureStr}"
      }, 1000)
      fileUtil.writeLinesToFile(midFeatureFile, midFeatures)
    }

    {
      println("Computing MID pair features")
      val midPairFeatures = fileUtil.parMapLinesFromFile(midPairFile, (line: String) => {
        val midPair = line
        val midPairFields = midPair.split(" ")
        val mid1 = midPairFields(0)
        val mid2 = midPairFields(1)
        val features = computer.computeMidPairFeatures(mid1, mid2)
        val featureStr = features.mkString(" -#- ")
        s"${midPair}\t${featureStr}"
      }, 1000)
      fileUtil.writeLinesToFile(midPairFeatureFile, midPairFeatures)
    }
  }

  def processMemoryConstrained() {
    val fileUtil = new FileUtil
    val computer = new PreFilteredFeatureComputer(sfeSpecFile, fileUtil)
    val midWriter = fileUtil.getFileWriter(midFeatureFile)
    fileUtil.parProcessFileInChunks(midFile, (lines: Seq[String]) => {
      val midFeatures = lines.map(line => {
        val mid = line
        val features = computer.computeMidFeatures(mid)
        val featureStr = features.mkString(" -#- ")
        s"${mid}\t${featureStr}\n"
      })
      midWriter synchronized {
        for (midFeatureLine <- midFeatures) {
          midWriter.write(midFeatureLine)
        }
      }
    })
    midWriter.close()

    val midPairWriter = fileUtil.getFileWriter(midPairFeatureFile)
    fileUtil.parProcessFileInChunks(midPairFile, (lines: Seq[String]) => {
      val midPairFeatures = lines.map(line => {
        val midPair = line
        val midPairFields = midPair.split(" ")
        val mid1 = midPairFields(0)
        val mid2 = midPairFields(1)
        val features = computer.computeMidPairFeatures(mid1, mid2)
        val featureStr = features.mkString(" -#- ")
        s"${midPair}\t${featureStr}\n"
      })
      midPairWriter synchronized {
        for (midPairFeatureLine <- midPairFeatures) {
          midPairWriter.write(midPairFeatureLine)
        }
      }
    })
    midPairWriter.close()
  }
}

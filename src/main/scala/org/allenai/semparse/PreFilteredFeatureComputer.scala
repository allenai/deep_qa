package org.allenai.semparse

import edu.cmu.ml.rtw.pra.experiments.Outputter
import edu.cmu.ml.rtw.pra.experiments.RelationMetadata
import edu.cmu.ml.rtw.pra.data.NodeInstance
import edu.cmu.ml.rtw.pra.data.NodePairInstance
import edu.cmu.ml.rtw.pra.features.NodePairSubgraphFeatureGenerator
import edu.cmu.ml.rtw.pra.features.NodeSubgraphFeatureGenerator
import edu.cmu.ml.rtw.pra.graphs.Graph
import edu.cmu.ml.rtw.pra.graphs.Graph
import edu.cmu.ml.rtw.users.matt.util.FileUtil
import edu.cmu.ml.rtw.users.matt.util.SpecFileReader

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

object compute_full_feature_vectors {
  val dataSize = "small"
  val midFile = s"data/${dataSize}/training-mids.tsv"
  val midPairFile = s"data/${dataSize}/training-mid-pairs.tsv"
  val midFeatureFile = s"data/${dataSize}/pre-filtered-mid-features.tsv"
  val midPairFeatureFile = s"data/${dataSize}/pre-filtered-mid-pair-features.tsv"
  val specFile = s"src/main/resources/sfe_spec_${dataSize}.json"

  def main(args: Array[String]) {
    processInMemory()
  }

  def processInMemory() {
    val fileUtil = new FileUtil
    val computer = new PreFilteredFeatureComputer(specFile, fileUtil)

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
    val computer = new PreFilteredFeatureComputer(specFile, fileUtil)
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

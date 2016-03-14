package org.allenai.semparse

import com.jayantkrish.jklol.tensor.Tensor

import edu.cmu.ml.rtw.pra.experiments.Outputter
import edu.cmu.ml.rtw.pra.experiments.RelationMetadata
import edu.cmu.ml.rtw.pra.data.NodeInstance
import edu.cmu.ml.rtw.pra.data.NodePairInstance
import edu.cmu.ml.rtw.pra.features.NodePairSubgraphFeatureGenerator
import edu.cmu.ml.rtw.pra.features.NodeSubgraphFeatureGenerator
import edu.cmu.ml.rtw.pra.graphs.Graph
import edu.cmu.ml.rtw.pra.graphs.Graph
import com.mattg.util.FileUtil
import com.mattg.util.SpecFileReader

import org.json4s._

import scala.collection.mutable

class SfeFeatureComputer(specFile: String, fileUtil: FileUtil = new FileUtil) {
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

  // And here we set up the cached entity feature files and pre-computed word feature dictionaries.
  val midFeatureFile = (params \ "mid feature file").extract[String]
  val midPairFeatureFile = (params \ "mid pair feature file").extract[String]
  val catWordFeatureFile = (params \ "cat word feature file").extract[String]
  val relWordFeatureFile = (params \ "rel word feature file").extract[String]

  val midFeatures = new CachedFeatureDictionary(midFeatureFile, catWordFeatureFile) {
    override def computeFeaturesForKey(key: String) = computeMidFeatures(key)
  }

  val midPairFeatures = new CachedFeatureDictionary(midPairFeatureFile, relWordFeatureFile) {
    override def computeFeaturesForKey(key: String) = computeMidPairFeatures(key)
  }

  val catWordFeatureMap = new FeatureMap(catWordFeatureFile)
  val relWordFeatureMap = new FeatureMap(relWordFeatureFile)

  val defaultWordFeatures = Seq("bias")

  // This has to match the separator used in the midPairFeatureFile, or the cache will not work.
  val entitySeparator = " "

  def getFeaturesForCatWord(word: String): Seq[String] =
    catWordFeatureMap.getFeatures(word, defaultWordFeatures)

  def getFeaturesForRelWord(word: String): Seq[String] =
    relWordFeatureMap.getFeatures(word, defaultWordFeatures)

  def getEntityFeatures(entity: String, word: String): Tensor =
    midFeatures.getFeatureVector(entity, word)

  def getEntityPairFeatures(entity1: String, entity2: String, word: String): Tensor =
    midPairFeatures.getFeatureVector(makeEntityPairKey(entity1, entity2), word)

  def makeEntityPairKey(entity1: String, entity2: String) = {
    entity1 + entitySeparator + entity2
  }

  def splitEntityPairKey(mid_pair: String) = {
    val fields = mid_pair.split(entitySeparator)
    (fields(0), fields(1))
  }

  def computeMidFeatures(mid: String): Seq[String] = {
    val bias = midFeatures.biasFeature
    val allowedFeatures = catWordFeatureMap.allAllowedFeatures
    if (graph.hasNode(mid)) {
      println(s"Computing feature vector for entity $mid")
      val instance = new NodeInstance(graph.getNodeIndex(mid), true, graph)
      val subgraph = nodeFeatureGenerator.getLocalSubgraph(instance)
      val features = nodeFeatureGenerator.extractFeaturesAsStrings(instance, subgraph)
      allowedFeatures.intersect(features.toSet).toSeq :+ bias
    } else {
      Seq(bias)
    }
  }

  def computeMidPairFeatures(mid_pair: String): Seq[String] = {
    val (mid1, mid2) = splitEntityPairKey(mid_pair)
    val bias = midPairFeatures.biasFeature
    val allowedFeatures = relWordFeatureMap.allAllowedFeatures
    if (graph.hasNode(mid1) && graph.hasNode(mid2)) {
      println(s"Computing feature vector for entity pair ($mid1, $mid2)")
      val instance = new NodePairInstance(graph.getNodeIndex(mid1), graph.getNodeIndex(mid2), true, graph)
      val subgraph = nodePairFeatureGenerator.getLocalSubgraph(instance)
      val features = nodePairFeatureGenerator.extractFeaturesAsStrings(instance, subgraph)
      allowedFeatures.intersect(features.toSet).toSeq :+ bias
    } else {
      Seq(bias)
    }
  }

  // This is only used at test time, on a relatively small set of entities.  Some of them repeat a
  // few times, though, and for hub entities, this can be an expensive computation.  So we're going
  // to cache the results of the findRelatedEntities computation.
  //
  // And, we currently ignore the word when finding related entities (we just look for direct (or
  // mediated) connections), so we just have the mid as the key here, instead of (word, mid).
  val cachedRelatedEntities = new mutable.HashMap[String, Set[String]]

  def findRelatedEntities(word: String, mid: String, isSource: Boolean): Set[String] = {
    cachedRelatedEntities.get(mid) match {
      case Some(related) => related
      case None => {
        println(s"Getting entities related to $mid by word $word")
        val related = nodePairFeatureGenerator.getRelatedNodes(mid, isSource, Seq("CONNECTED"), graph)
        val kept = related.filter(_.startsWith("/m/"))  // Only keep actual entities, not type nodes.
        println(s"Found ${kept.size} of them")
        cachedRelatedEntities(mid) = kept
        kept
      }
    }
  }
}

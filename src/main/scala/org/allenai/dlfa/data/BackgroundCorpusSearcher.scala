package org.allenai.dlfa.data

import java.net.InetSocketAddress
import org.apache.commons.lang3.StringUtils
import org.elasticsearch.client.transport.TransportClient
import org.elasticsearch.common.settings.Settings
import org.elasticsearch.common.transport.InetSocketTransportAddress
import org.elasticsearch.index.query.QueryBuilders

import com.mattg.util.JsonHelper

import scala.collection.mutable

import org.json4s._

abstract class BackgroundCorpusSearcher(params: JValue) {
  val baseParams = Seq("searcher type", "num passages per query")

  val numPassagesPerQuery = JsonHelper.extractWithDefault(params, "num passages per query", 10)

  def getBackground(query: String): Seq[String]
}

object BackgroundCorpusSearcher {
  def create(params: JValue): BackgroundCorpusSearcher = {
    val searcherType = JsonHelper.extractWithDefault(params, "searcher type", "lucene")
    searcherType match {
      case "lucene" => new LuceneBackgroundCorpusSearcher(params)
      case _ => throw new IllegalStateException(s"Unrecognized background searcher type: $searcherType")
    }
  }
}

class LuceneBackgroundCorpusSearcher(params: JValue) extends BackgroundCorpusSearcher(params) {
  implicit val formats = DefaultFormats
  val name = "Lucene Background Knowledge Searcher"

  val validParams = baseParams ++ Seq(
    "max sentence length",
    "min sentence length",
    "hit multiplier",
    "remove query from results",
    "remove query near duplicates",
    "elastic search index url",
    "elastic search index name",
    "elastic search cluster name",
    "elastic search index port"
  )
  JsonHelper.ensureNoExtras(params, name, validParams)

  val maxSentenceLength = JsonHelper.extractWithDefault(params, "max sentence length", 100)
  val minSentenceLength = JsonHelper.extractWithDefault(params, "min sentence length", 3)

  // We get hitMultiplier times as many results from Lucene as we're looking for, so we can filter
  // through them and discard duplicates, ones that are too short, or other filtering.
  val hitMultiplier = JsonHelper.extractWithDefault(params, "hit multiplier", 5)

  // If the sentences we're querying with are drawn from the same corpus as we are querying with
  // Lucene, the query sentence will always be returned when we do a search.  This is pretty bad
  // for producing training data, so we remove the query from the returned results.  However, at
  // test time, if you happen to have the query sentence in the results, that's not a problem, so
  // we make this a parameter.
  val removeQuery = JsonHelper.extractWithDefault(params, "remove query from results", true)

  // When you've created negative data from positive data in the corpus we're querying, you don't
  // really want the positive sentence to be returned for the negative sentence query.
  val removeQueryNearDuplicates = JsonHelper.extractWithDefault(params, "remove query near duplicates", false)

  val esUrl = JsonHelper.extractWithDefault(params, "elastic search index url", "aristo-es1.dev.ai2")
  val esPort = JsonHelper.extractWithDefault(params, "elastic search index port", 9300)
  val esClusterName = JsonHelper.extractWithDefault(params, "elastic search cluster name", "aristo-es")
  val esIndexName = JsonHelper.extractWithDefault(params, "elastic search index name", "busc")

  lazy val address = new InetSocketTransportAddress(new InetSocketAddress(esUrl, esPort))
  lazy val settings = Settings.builder().put("cluster.name", esClusterName).build()
  lazy val esClient = TransportClient.builder().settings(settings).build().addTransportAddress(address)

  override def getBackground(query: String): Seq[String] = {
    val response = esClient.prepareSearch(esIndexName)
      .setTypes("sentence")
      .setQuery(QueryBuilders.matchQuery("text", query))
      .setFrom(0).setSize(numPassagesPerQuery * hitMultiplier).setExplain(true)
      .execute()
      .actionGet()
    val passages = response.getHits().getHits().map(hit => {
      hit.sourceAsMap().get("text").asInstanceOf[String]
    })
    consolidateHits(query, passages, numPassagesPerQuery)
  }

  def consolidateHits(query: String, hits: Seq[String], maxToKeep: Int): Seq[String] = {
    val kept = new mutable.HashSet[String]
    var i = 0
    while (kept.size < maxToKeep && i < hits.size) {
      // Unicode character 0085 is a "next line" character that isn't caught by \s for some reason.
      val hit = hits(i).replaceAll("\\s+", " ").replace("\u0085", " ")
      if (shouldKeep(query, hit, kept.toSet)) {
        kept += hit
      }
      i += 1
    }
    kept.toSeq
  }

  private def shouldKeep(query: String, hit: String, keptSoFar: Set[String]): Boolean = {
    if (removeQuery && hit == query) return false
    if (removeQueryNearDuplicates) {
      val queryWords = query.split(" ").map(_.toLowerCase).toSet
      val hitWords = hit.split(" ").map(_.toLowerCase).toSet
      if ((hitWords -- queryWords).size <= 1) return false
    }
    if (keptSoFar.contains(hit)) return false
    val sentenceLength = hit.split(" ").length
    if (sentenceLength < minSentenceLength) return false
    if (sentenceLength > maxSentenceLength) return false

    // Jaro-Winkler distance is essentially edit distance / number of characters.
    if (keptSoFar.exists(kept => StringUtils.getJaroWinklerDistance(kept, hit) > .8)) return false
    return true
  }
}

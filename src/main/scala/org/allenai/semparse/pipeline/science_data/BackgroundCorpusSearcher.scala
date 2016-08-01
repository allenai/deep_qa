package org.allenai.semparse.pipeline.science_data

import org.json4s._

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.elasticsearch.client.transport.TransportClient
import org.elasticsearch.common.settings.Settings
import org.elasticsearch.common.transport.InetSocketTransportAddress
import org.elasticsearch.index.query.QueryBuilders

import org.apache.commons.lang3.StringUtils

import scala.collection.mutable

import java.net.InetSocketAddress

/**
 * A BackgroundCorpusSearcher takes as input a list of sentences and some kind of background
 * corpus, then produces as output (for each sentence) a list of passages from the background
 * corpus that are relevant to the input sentence.  We say "passages" here, because the retrieved
 * background information could be sentences, paragraphs, snippets, or something else.
 *
 * Basic input/output spec for subclasses: sentences should be encoded one sentence per line as
 * "[sentence index][tab][sentence]"; output should be encoded one sentence per line as
 * "[sentence index][tab][background passage 1][tab][background passage 2][tab][...]".  The
 * subclass can determine what other inputs/outputs it needs, but should adhere to this basic spec.
 *
 * See note in SentenceProducer for why this is a trait instead of an abstract class.  Basically,
 * this is necessary to allow some subclasses to be SubprocessSteps instead of just Steps.
 */
trait BackgroundCorpusSearcher {
  def params: JValue
  def fileUtil: FileUtil

  val baseParams = Seq("sentences", "num passages per sentence")

  val numPassagesPerSentence = JsonHelper.extractWithDefault(params, "num passages per sentence", 10)

  val sentenceProducer = SentenceProducer.create(params \ "sentences", fileUtil)
  if (!sentenceProducer.indexSentences) {
    throw new IllegalStateException("background corpus search needs indexed sentences!")
  }
  val sentencesFile = sentenceProducer.outputFile

  val sentencesInput: (String, Option[Step]) = (sentencesFile, Some(sentenceProducer))

  val outputFile = sentencesFile.dropRight(4) + "_background.tsv"

  def outputBackground(backgroundPassages: Seq[(Int, Seq[String])]) {
    val lines = backgroundPassages.map(background => {
      val (index, passages) = background
      val passagesStr = passages.mkString("\t")
      s"${index}\t${passagesStr}"
    })
    fileUtil.writeLinesToFile(outputFile, lines)
  }
}

object BackgroundCorpusSearcher {
  def create(params: JValue, fileUtil: FileUtil): Step with BackgroundCorpusSearcher = {
    new LuceneBackgroundCorpusSearcher(params, fileUtil)
  }
}

/**
 * This is a BackgroundCorpusSearcher that uses Lucene to find relevant sentences from the
 * background corpus.  In addition to the basic input sentence file, we need to know where the
 * Elastic Search index is that we should send queries to.
 *
 * In this implementation, all background passages are sentences.  To change that while still using
 * Lucene, use a different document type (see the `setType("sentence")` line below).  Note that
 * you'll have to create that document type when you generate the index.
 */
class LuceneBackgroundCorpusSearcher(
  val params: JValue,
  val fileUtil: FileUtil
) extends Step(Some(params), fileUtil) with BackgroundCorpusSearcher {
  implicit val formats = DefaultFormats
  override val name = "Lucene Background Knowledge Searcher"

  val validParams = baseParams ++ Seq(
    "max sentence length",
    "min sentence length",
    "hit multiplier",
    "remove query from results",
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

  val esUrl = JsonHelper.extractWithDefault(params, "elastic search index url", "aristo-es1.dev.ai2")
  val esPort = JsonHelper.extractWithDefault(params, "elastic search index port", 9300)
  val esClusterName = JsonHelper.extractWithDefault(params, "elastic search cluster name", "aristo-es")
  val esIndexName = JsonHelper.extractWithDefault(params, "elastic search index name", "busc")

  override val inputs: Set[(String, Option[Step])] = Set(sentencesInput)
  override val outputs: Set[String] = Set(outputFile)
  override val paramFile = outputFile.dropRight(4) + "_params.json"
  override val inProgressFile = outputFile.dropRight(4) + "_in_progress"

  override def _runStep() {
    val address = new InetSocketTransportAddress(new InetSocketAddress(esUrl, esPort))
    val settings = Settings.builder().put("cluster.name", esClusterName).build()
    val esClient = TransportClient.builder().settings(settings).build().addTransportAddress(address)

    // TODO(matt): might want to make this streaming, for large input files
    val lines = fileUtil.readLinesFromFile(sentencesFile)
    val indexedSentences = lines.map(line => {
      val fields = line.split("\t")
      (fields(0).toInt, fields(1))
    })
    val backgroundPassages = indexedSentences.par.map(indexedSentence => {
      val (index, sentence) = indexedSentence
      val response = esClient.prepareSearch(esIndexName)
        .setTypes("sentence")
        .setQuery(QueryBuilders.matchQuery("text", sentence))
        .setFrom(0).setSize(numPassagesPerSentence * hitMultiplier).setExplain(true)
        .execute()
        .actionGet()
      val passages = response.getHits().getHits().map(hit => {
        hit.sourceAsMap().get("text").asInstanceOf[String]
      })
      val keptPassages = consolidateHits(sentence, passages, numPassagesPerSentence)
      (index, keptPassages)
    }).seq
    outputBackground(backgroundPassages)
  }

  def consolidateHits(query: String, hits: Seq[String], maxToKeep: Int): Seq[String] = {
    val kept = new mutable.ArrayBuffer[String]
    var i = 0
    while (kept.size < maxToKeep && i < hits.size) {
      // Unicode character 0085 is a "next line" character that isn't caught by \s for some reason.
      val hit = hits(i).replaceAll("\\s+", " ").replace("\u0085", " ")
      if (shouldKeep(query, hit, kept)) {
        kept += hit
      }
      i += 1
    }
    kept
  }

  private def shouldKeep(query: String, hit: String, keptSoFar: Seq[String]): Boolean = {
    if (removeQuery && hit == query) return false
    if (keptSoFar.contains(hit)) return false
    val sentenceLength = hit.split(" ").length
    if (sentenceLength < minSentenceLength) return false
    if (sentenceLength > maxSentenceLength) return false

    // Jaro-Winkler distance is essentially edit distance / number of characters.
    if (keptSoFar.exists(kept => StringUtils.getJaroWinklerDistance(kept, hit) > .8)) return false
    return true
  }
}

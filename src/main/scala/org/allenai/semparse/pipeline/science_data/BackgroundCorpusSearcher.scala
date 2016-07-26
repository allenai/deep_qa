package org.allenai.semparse.pipeline.science_data

import org.json4s._

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.elasticsearch.client.transport.TransportClient
import org.elasticsearch.common.settings.Settings
import org.elasticsearch.common.transport.InetSocketTransportAddress
import org.elasticsearch.index.query.QueryBuilders

import java.net.InetSocketAddress

/**
 * A BackgroundCorpusSearcher takes as input a list of sentences and some kind of background
 * corpus, then produces as output (for each sentence) a list of passages from the background
 * corpus that are relevant to the input sentence.
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

  val baseParams = Seq("sentences")
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
 */
class LuceneBackgroundCorpusSearcher(
  val params: JValue,
  val fileUtil: FileUtil
) extends Step(Some(params), fileUtil) with BackgroundCorpusSearcher {
  implicit val formats = DefaultFormats
  override val name = "Lucene Background Knowledge Searcher"

  val validParams = baseParams ++ Seq("num passages per sentence", "elastic search index url",
    "elastic search index name", "elastic search cluster name", "elastic search index port")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val numPassagePerSentence = JsonHelper.extractWithDefault(params, "num passages per sentence", 10)
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
    val lines = fileUtil.readLinesFromFile(sentencesFile).take(100)  // TODO(matt): REMOVE THE TAKE!
    val indexedSentences = lines.map(line => {
      val fields = line.split("\t")
      (fields(0).toInt, fields(1))
    })
    val backgroundPassages = indexedSentences.par.map(indexedSentence => {
      val (index, sentence) = indexedSentence
      val response = esClient.prepareSearch(esIndexName)
        .setQuery(QueryBuilders.matchQuery("text", sentence))
        .setFrom(0).setSize(numPassagePerSentence).setExplain(true)
        .execute()
        .actionGet()
      val passages = response.getHits().getHits().map(hit => {
        hit.sourceAsMap().get("text").asInstanceOf[String]
      })
      (index, passages.toSeq)
    }).seq
    outputBackground(backgroundPassages)
  }
}

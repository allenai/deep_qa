package org.allenai.deep_qa.pipeline

import java.net.InetSocketAddress
import org.apache.commons.lang3.StringUtils
import org.elasticsearch.client.transport.TransportClient
import org.elasticsearch.common.settings.Settings
import org.elasticsearch.common.transport.InetSocketTransportAddress
import org.elasticsearch.index.query.QueryBuilders

import scala.collection.mutable

import org.json4s._

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.allenai.deep_qa.data.BackgroundCorpusSearcher


/**
 * A BackgroundCorpusSearcherStep takes as input a list of sentences and some kind of background
 * corpus, then produces as output (for each sentence) a list of passages from the background
 * corpus that are relevant to the input sentence.  We say "passages" here, because the retrieved
 * background information could be sentences, paragraphs, snippets, or something else.
 *
 * Most of the work for this Step is done (unsurprisingly) by a BackgroundCorpusSearcher.  This
 * class just has code for reading an input file, passing off the input to a Searcher, and
 * outputting the result.
 *
 * Basic input/output spec for subclasses: sentences should be encoded one sentence per line as
 * "[sentence index][tab][sentence]"; output should be encoded one sentence per line as
 * "[sentence index][tab][background passage 1][tab][background passage 2][tab][...]".  The
 * subclass can determine what other inputs/outputs it needs, but should adhere to this basic spec.
 *
 * TODO(matt): we actually allow a couple of different formats now.  It'd be better to have some
 * kind of DatasetReader, with type specified by the previous Step, and then use an Instance object
 * to create a query...
 *
 * See note in SentenceProducer for why this is a trait instead of an abstract class.  Basically,
 * this is necessary to allow some subclasses to be SubprocessSteps instead of just Steps.
 */
trait BackgroundCorpusSearcherStep {
  def params: JValue
  def fileUtil: FileUtil

  val baseParams = Seq("type", "searcher", "sentences", "sentence format")

  lazy val searcher = BackgroundCorpusSearcher.create(params \ "searcher")

  lazy val sentenceProducer = {
    val producer = SentenceProducer.create(params \ "sentences", fileUtil)
    if (!producer.indexSentences) {
      throw new IllegalStateException("background corpus search needs indexed sentences!")
    }
    producer
  }
  lazy val sentencesFile = sentenceProducer.outputFile

  lazy val sentencesInput: (String, Option[Step]) = (sentencesFile, Some(sentenceProducer))

  lazy val outputFile = sentencesFile.dropRight(4) + "_background.tsv"
}

object BackgroundCorpusSearcherStep {
  def create(params: JValue, fileUtil: FileUtil): Step with BackgroundCorpusSearcherStep = {
    val searcherType = JsonHelper.extractWithDefault(params, "type", "default")
    searcherType match {
      case "default" => new DefaultBackgroundCorpusSearcherStep(params, fileUtil)
      case "manually provided" => new ManuallyProvidedBackground(params, fileUtil)
      case _ => throw new IllegalStateException(s"Unrecognized background searcher step type: $searcherType")
    }
  }
}


class DefaultBackgroundCorpusSearcherStep(
  val params: JValue,
  val fileUtil: FileUtil
) extends Step(Some(params), fileUtil) with BackgroundCorpusSearcherStep {
  override val name = "Background Corpus Searcher Step"
  val validParams = baseParams
  JsonHelper.ensureNoExtras(params, name, validParams)

  // TODO(matt): Is there a way to get this from the sentence producer, instead of from a param?
  val formatChoices = Seq("plain sentence", "question and answer")
  val sentenceFormat = JsonHelper.extractChoiceWithDefault(
    params,
    "sentence format",
    formatChoices,
    formatChoices(0)
  )

  override val inputs: Set[(String, Option[Step])] = Set(sentencesInput)
  override val outputs: Set[String] = Set(outputFile)
  override val paramFile = outputFile.dropRight(4) + "_params.json"
  override val inProgressFile = outputFile.dropRight(4) + "_in_progress"

  override def _runStep() {
    // TODO(matt): might want to make this streaming, for large input files
    val lines = fileUtil.readLinesFromFile(sentencesFile)
    val indexedSentences = lines.map(line => {
      val fields = line.split("\t")
      sentenceFormat match {
        case "plain sentence" => (fields(0).toInt, Seq(fields(1)))
        case "question and answer" => {
          val answers = fields(2).split("###")
          val queries = answers.map(fields(1) + " " + _).toSeq
          (fields(0).toInt, Seq(fields(1)) ++ queries)
        }
      }
    })
    val backgroundPassages = indexedSentences.par.map(indexedSentence => {
      val (index, queries) = indexedSentence
      val keptPassages = queries.flatMap(query => {
        searcher.getBackground(query)
      }).toSet.toSeq
      (index, keptPassages)
    }).seq
    outputBackground(backgroundPassages)
  }

  def outputBackground(backgroundPassages: Seq[(Int, Seq[String])]) {
    val lines = backgroundPassages.map(background => {
      val (index, passages) = background
      val passagesStr = passages.mkString("\t")
      s"${index}\t${passagesStr}"
    })
    fileUtil.writeLinesToFile(outputFile, lines)
  }
}

/**
 * This BackgroundCorpusSearcherStep lets you manually override the pipeline, giving a background
 * file that is not generated by one of the steps here.  In general, this should be used sparingly,
 * mostly for testing or while things are still in development, as it kind of defeats the whole
 * purpose of the pipeline code.
 */
class ManuallyProvidedBackground(
  val params: JValue,
  val fileUtil: FileUtil
) extends Step(Some(params), fileUtil) with BackgroundCorpusSearcherStep {
  implicit val formats = DefaultFormats
  override val name = "Manually Provided Background"

  val validParams = baseParams ++ Seq("filename")
  JsonHelper.ensureNoExtras(params, name, validParams)

  override lazy val outputFile = (params \ "filename").extract[String]
  override val inputs: Set[(String, Option[Step])] = Set((outputFile, None))
  override val outputs = Set(outputFile)
  override val inProgressFile = outputFile.dropRight(4) + "_in_progress"

  override def _runStep() { }
}

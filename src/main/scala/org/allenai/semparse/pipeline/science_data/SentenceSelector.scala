package org.allenai.semparse.pipeline.science_data

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.json4s._

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import org.allenai.semparse.parse.Parser

/**
 * The job of this Step is to select sentences from a large corpus that are suitable for training a
 * deep neural network.  At present, this is very simple, selecting only based on the number of
 * words in the sentence, but the plan is to make this much more complicated eventually, likely
 * involved a trained classifier, a coreference resolution system, or other things.
 *
 * INPUTS: a collection of documents
 * OUTPUTS: a file containing sentences, one sentence per line.  File format is
 * "[sentence index][tab][sentence]", or just "[sentence]", depending on the `use sentence indices`
 * parameter.
 */
class SentenceSelectorStep(
  val params: JValue,
  val fileUtil: FileUtil
) extends Step(Some(params), fileUtil) with SentenceProducer {
  implicit val formats = DefaultFormats
  override val name = "Sentence Selector Step"
  val validParams = baseParams ++ Seq("sentence selector", "data directory", "data name")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val dataName = (params \ "data name").extract[String]
  val dataDir = (params \ "data directory").extract[String]

  val filename = if (indexSentences) "indexed_training_data.tsv" else "training_data.tsv"
  override val outputFile = s"data/science/$dataName/$filename"
  val numPartitions = 1

  // The check for required inputs can't handle files on S3, so if the given data directory is on
  // S3, we just tell the pipeline code that there is no input, and we'll fail in another way if
  // the S3 file doesn't actually exist.
  override val inputs: Set[(String, Option[Step])] =
    if (dataDir.startsWith("s3n://")) Set() else Set((dataDir, None))
  override val outputs = Set(outputFile)
  override val paramFile = outputs.head.dropRight(4) + "_params.json"
  override val inProgressFile = outputs.head.dropRight(4) + "_in_progress"

  val sentenceSelector = new SentenceSelector(params \ "sentence selector")

  override def _runStep() {
    val conf = new SparkConf().setAppName(s"Sentence to Logic")
      .set("spark.driver.maxResultSize", "0")
      .set("spark.network.timeout", "100000")
      .set("spark.akka.frameSize", "1028")
      .setMaster("local[*]")

    val sc = new SparkContext(conf)

    val sentences = processCorpus(sc, dataDir, numPartitions, sentenceSelector)
    outputSentences(sentences)
    sc.stop()
  }

  def processCorpus(
    sc: SparkContext,
    dataDir: String,
    numPartitions: Int,
    sentenceSelector: SentenceSelector
  ): Seq[String] = {
    val sentences = sc.textFile(dataDir, numPartitions).flatMap(line => {
      val tagsRemoved = line.replace("<SENT>", "").replace("</SENT>", "")
      val sentences = Parser.stanford.splitSentences(tagsRemoved)
      sentences.filter(sentenceSelector.shouldKeepSentence)
    }).distinct()
    sentences.collect()
  }
}

class SentenceSelector(params: JValue) extends Serializable {
  val validParams = Seq("min word count per sentence", "max word count per sentence")
  val minWordCount = JsonHelper.extractWithDefault(params, "min word count per sentence", 4)
  val maxWordCount = JsonHelper.extractWithDefault(params, "max word count per sentence", 20)
  JsonHelper.ensureNoExtras(params, "Sentence Selector", validParams)

  val badChars = Seq("?", "!", ":", ";", "&", "_", "-", "\\", "(", ")", "{", "}", "[", "]", "<", ">", "\"", "'", "=", "|", "~", "%")

  def shouldKeepSentence(sentence: String): Boolean = {
    val wordCount = sentence.split(" ").length
    if (wordCount < minWordCount) return false
    if (wordCount > maxWordCount) return false
    for (char <- badChars) {
      if (sentence.contains(char)) return false
    }
    if (hasPronoun(sentence)) return false
    return true
  }

  val pronouns = Seq("i", "me", "my", "we", "us", "our", "you", "your", "it", "its", "he", "him",
    "his", "she", "her", "hers", "they", "them", "this", "these")
  def hasPronoun(sentence: String): Boolean = {
    val lower = sentence.toLowerCase
    for (pronoun <- pronouns) {
      if (lower.startsWith(pronoun + " ") ||
          lower.contains(" " + pronoun + " ") ||
          lower.endsWith(" " + pronoun + ".")) {
            return true
      }
    }
    return false
  }
}

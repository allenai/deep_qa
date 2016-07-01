package org.allenai.semparse.pipeline.science_data

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.json4s._

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

/**
 * The job of this Step is to select sentences from a large corpus that are suitable for training a
 * deep neural network.  At present, this is very simple, selecting only based on the number of
 * words in the sentence, but the plan is to make this much more complicated eventually, likely
 * involved a trained classifier, a coreference resolution system, or other things.
 */
class SentenceSelectorStep(
  params: JValue,
  fileUtil: FileUtil
) extends Step(Some(params), fileUtil) {
  implicit val formats = DefaultFormats
  override val name = "Sentence Selector Step"
  val validParams = Seq("sentence selector", "data directory", "data name")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val dataName = (params \ "data name").extract[String]
  val dataDir = (params \ "data directory").extract[String]

  val outputFile = s"data/science/$dataName/training_data.tsv"
  val numPartitions = 1

  override val inputs: Set[(String, Option[Step])] = Set((dataDir, None))
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

    val sentences = sc.textFile(dataDir, numPartitions).flatMap(line => {
      val sentence = line.replace("<SENT>", "").replace("</SENT>", "")
      if (sentenceSelector.shouldKeepSentence(line)) Seq(sentence) else Seq()
    }).distinct()
    val outputLines = sentences.collect()
    fileUtil.writeLinesToFile(outputFile, outputLines)

    sc.stop()
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
          lower.endsWith(" " + pronoun + "."))
        return true
    }
    return false
  }
}

package org.allenai.semparse.pipeline.science_data

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.json4s._

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import org.allenai.semparse.parse.DependencyTree
import org.allenai.semparse.parse.LogicalFormGenerator
import org.allenai.semparse.parse.Predicate
import org.allenai.semparse.parse.StanfordParser

class ScienceSentenceProcessor(
  params: JValue,
  fileUtil: FileUtil
) extends Step(Some(params), fileUtil) {
  implicit val formats = DefaultFormats
  override val name = "Science Sentence Processor"

  val validParams = Seq("min word count per sentence", "max word count per sentence",
    "data directory", "data name", "output format")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val dataName = (params \ "data name").extract[String]
  val dataDir = (params \ "data directory").extract[String]
  val minWordCount = JsonHelper.extractWithDefault(params, "min word count per sentence", 4)
  val maxWordCount = JsonHelper.extractWithDefault(params, "max word count per sentence", 20)
  val formatOptions = Seq("training data", "debug")
  val outputFormat = JsonHelper.extractOptionWithDefault(params, "output format", formatOptions, "training data")
  val outputFile = outputFormat match {
    case "training data" => s"data/science/$dataName/training_data.tsv"
    case "debug" => s"data/science/$dataName/sentence_lfs.txt"
  }

  val numPartitions = 1

  override val inputs: Set[(String, Option[Step])] = Set((dataDir, None))
  override val outputs = Set(outputFile)
  override val paramFile = outputs.head.dropRight(4) + "_params.json"
  override val inProgressFile = outputs.head.dropRight(4) + "_in_progress"

  override def _runStep() {
    val conf = new SparkConf().setAppName(s"Compute PMI")
      .set("spark.driver.maxResultSize", "0")
      .set("spark.network.timeout", "1000000")
      .set("spark.akka.frameSize", "1028")
      .setMaster("local[*]")

    val sc = new SparkContext(conf)

    parseSentences(
      sc,
      outputFile,
      dataDir,
      minWordCount,
      maxWordCount,
      outputFormat
    )
  }

  def parseSentences(
    sc: SparkContext,
    outputFile: String,
    dataDir: String,
    minWordCount: Int,
    maxWordCount: Int,
    outputFormat: String
  ) {
    fileUtil.mkdirsForFile(outputFile)
    val sentences = sc.textFile(dataDir, numPartitions).flatMap(line => {
      val sentence = line.replace("<SENT>", "").replace("</SENT>", "")
      if (Helper.shouldKeepSentence(sentence, minWordCount, maxWordCount)) {
        Seq(sentence)
      } else {
        Seq()
      }
    }).distinct()
    val trees = sentences.flatMap(Helper.parseSentence)
    val logicalForms = trees.map(sentenceAndTree => {
      val sentence = sentenceAndTree._1
      val tree = sentenceAndTree._2
      val logicalForm = try {
        LogicalFormGenerator.getLogicalForm(tree)
      } catch {
        case e: Throwable => { println(sentence); tree.print(); throw e }
      }
      (sentence, logicalForm)
    })
    val outputStrings = logicalForms.flatMap(Helper.logicalFormToString(outputFormat))

    val finalOutput = outputStrings.collect()
    fileUtil.writeLinesToFile(outputFile, finalOutput)
  }
}

// This semi-ugliness is so that the spark functions are serializable.
object Helper {
  val parser = new StanfordParser

  val badChars = Seq("?", "!", ":", ";", "&", "_", "-", "\\", "(", ")", "{", "}", "[", "]", "<", ">", "\"", "'")
  def shouldKeepSentence(sentence: String, minWordCount: Int, maxWordCount: Int): Boolean = {
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

  def parseSentence(sentence: String) = {
    val parse = parser.parseSentence(sentence)
    parse.dependencyTree match {
      case None => Seq()
      case Some(tree) => {
        if (Helper.shouldKeepTree(tree)) {
          Seq((sentence, tree))
        } else {
          Seq()
        }
      }
    }
  }

  def shouldKeepTree(tree: DependencyTree): Boolean = {
    if (tree.token.posTag.startsWith("V")) {
      tree.getChildWithLabel("nsubj") match {
        case None => return false
        case _ => return true
      }
    }
    tree.getChildWithLabel("cop") match {
      case None => return false
      case _ => return true
    }
  }

  def logicalFormToString(
    outputFormat: String
  )(
    logicalForm: (String, Set[Predicate])
  ): Seq[String] = {
    outputFormat match {
      case "training data" => logicalFormToTrainingData(logicalForm)
      case "debug" => sentenceAndLogicalFormAsString(logicalForm)
      case _ => throw new IllegalStateException("Unrecognized output format")
    }
  }

  def logicalFormToTrainingData(logicalForm: (String, Set[Predicate])): Seq[String] = {
    val sentence = logicalForm._1
    val predicates = logicalForm._2
    val fullJson = "[" + predicates.map(predicate => {
      val names = predicate.arguments.map("\"" + _ + "\"").mkString(",")
      s"""["${predicate.predicate}",$names]"""
    }).mkString(", ") + "]"
    predicates.map(predicate => {
      // I'm trying to stay as close to the previous training data format as possible.  But, the
      // way the ids were formatted with Freebase data will not work for our noun phrases.  So, I'm
      // going to leave the id column blank as a single that the name column should be used
      // instead.
      val ids = ""
      val names = predicate.arguments.map("\"" + _ + "\"").mkString(" ")
      val catWord = if (predicate.arguments.size == 1) predicate.predicate else ""
      val relWord = if (predicate.arguments.size == 2) predicate.predicate else ""
      val wordRelOrCat = if (predicate.arguments.size == 1) "word-cat" else "word-rel"
      val lambdaExpression = s"""(($wordRelOrCat "${predicate.predicate}") $names)"""
      val fields = Seq(ids, names, catWord, relWord, lambdaExpression, fullJson, sentence)
      fields.mkString("\t")
    }).toSeq
  }

  def sentenceAndLogicalFormAsString(logicalForm: (String, Set[Predicate])): Seq[String] = {
    val sentence = logicalForm._1
    val lf = logicalForm._2
    val lfString = lf.map(_.toString).mkString(" ")
    Seq(s"${sentence} -> ${lfString}")
  }
}

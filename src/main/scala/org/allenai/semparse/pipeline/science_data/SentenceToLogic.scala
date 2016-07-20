package org.allenai.semparse.pipeline.science_data

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.json4s._

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import org.allenai.semparse.parse.Conjunction
import org.allenai.semparse.parse.DependencyTree
import org.allenai.semparse.parse.Logic
import org.allenai.semparse.parse.LogicalFormGenerator
import org.allenai.semparse.parse.Predicate
import org.allenai.semparse.parse.StanfordParser

import java.io.File

import scala.concurrent._
import scala.util.{Success, Failure}
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global

/**
 * Here we convert sentences into logical forms, using the LogicalFormGenerator.
 *
 * INPUTS: a file containing sentences, one sentence per line.  Expected file format is
 * "[sentence index][tab][sentence]", or just "[sentence]".
 *
 * OUTPUTS: a file containing logical statements, one statement per line.  Output file format is
 * the same as the input file, except sentences are replaced with logical forms.  We keep the same
 * index, if an index is provided.  Optionally, we drop a line if the logical form generation
 * failed for that line.
 */
class SentenceToLogic(
  params: JValue,
  fileUtil: FileUtil
) extends Step(Some(params), fileUtil) {
  implicit val formats = DefaultFormats
  override val name = "Sentence to Logic"

  val validParams = Seq("sentences", "logical forms", "output file", "drop errors")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val dropErrors = JsonHelper.extractWithDefault(params, "drop errors", true)

  val sentenceProducer = SentenceProducer.create(params \ "sentences", fileUtil)
  val sentencesFile = sentenceProducer.outputFile
  val logicalFormGenerator = new LogicalFormGenerator(params \ "logical forms")

  val outputFile = JsonHelper.extractAsOption[String](params, "output file") match {
    case None => sentencesFile.dropRight(4) + "_as_logic.tsv"
    case Some(filename) => filename
  }

  val numPartitions = 100

  override val inputs: Set[(String, Option[Step])] = Set((sentencesFile, Some(sentenceProducer)))
  override val outputs = Set(outputFile)
  override val paramFile = outputs.head.dropRight(4) + "_params.json"
  override val inProgressFile = outputs.head.dropRight(4) + "_in_progress"

  override def _runStep() {
    val conf = new SparkConf().setAppName(s"Sentence to Logic")
      .set("spark.driver.maxResultSize", "0")
      .set("spark.network.timeout", "100000")
      .set("spark.akka.frameSize", "1028")
      .setMaster("local[*]")

    val sc = new SparkContext(conf)

    parseSentences(sc, logicalFormGenerator, dropErrors, outputFile)

    sc.stop()
  }

  def parseSentences(
    sc: SparkContext,
    logicalFormGenerator: LogicalFormGenerator,
    dropErrors: Boolean,
    outputFile: String
  ) {
    fileUtil.mkdirsForFile(outputFile)
    val lines = sc.textFile(sentencesFile, numPartitions)
    val sentences = lines.map(SentenceToLogic.getSentenceFromLine)
    val trees = sentences.map(SentenceToLogic.parseSentence)
    val logicalForms = trees.flatMap(sentenceAndTree => {
      val result = SentenceToLogic.runWithTimeout(2000, () => {
        val tree = sentenceAndTree._2
        val logicalForm = try {
          tree.flatMap(logicalFormGenerator.getLogicalForm)
        } catch {
          case e: Throwable => { println(sentenceAndTree._1); tree.map(_.print()); throw e }
        }
        (sentenceAndTree._1, logicalForm)
      })
      result match {
        case None => {
          println(s"Timeout while processing sentence: ${sentenceAndTree._1._1}")
          if (dropErrors) Seq() else Seq((sentenceAndTree._1, None))
        }
        case Some(either) => either match {
          case Left(t) => {
            println(s"Exception thrown while processing sentence: ${sentenceAndTree._1._1} ---- ${t.getMessage}")
            if (dropErrors) Seq() else Seq((sentenceAndTree._1, None))
          }
          case Right(result) => Seq(result)
        }
      }
    })
    val outputStrings = logicalForms.flatMap(sentenceAndLf => {
      val result = SentenceToLogic.runWithTimeout(2000, () => {
        SentenceToLogic.sentenceAndLogicalFormAsString(sentenceAndLf)
      })
      val failureStr = sentenceAndLf._1._2.map(index => s"${index}\t").getOrElse("")
      result match {
        case None => {
          println(s"Timeout while printing sentence: ${sentenceAndLf._1}")
          if (dropErrors) Seq() else Seq(failureStr)
        }
        case Some(either) => either match {
          case Left(t) => {
            println(s"Exception thrown while printing sentence: ${sentenceAndLf._1} ---- ${t.getMessage}")
            if (dropErrors) Seq() else Seq(failureStr)
          }
          case Right(result) => if (result.isEmpty() && dropErrors) Seq() else Seq(result)
        }
      }
    })

    val finalOutput = outputStrings.collect()
    fileUtil.writeLinesToFile(outputFile, finalOutput)
  }
}

// This semi-ugliness is so that the spark functions are serializable.
object SentenceToLogic {
  type IndexedSentence = (String, Option[Int])

  val parser = new StanfordParser

  def runWithTimeout[T](milliseconds: Long, f: () => T): Option[Either[Throwable, T]] = {
    import scala.language.postfixOps
    val result = Future(f())
    try {
      Await.result(result, 1 seconds)
      result.value match {
        case None => None
        case Some(tryResult) => tryResult match {
          case Success(result) => Some(Right(result))
          case Failure(t) => {
            Some(Left(t))
          }
        }
      }
    } catch {
      case e: java.util.concurrent.TimeoutException => {
        None
      }
    }
  }

  def getSentenceFromLine(line: String): IndexedSentence = {
    val fields = line.split("\t")
    if (fields.length == 2 && fields(0).forall(Character.isDigit)) {
      (fields(1), Some(fields(0).toInt))
    } else {
      (fields(0), None)
    }
  }

  def parseSentence(indexedSentence: IndexedSentence): (IndexedSentence, Option[DependencyTree]) = {
    val parse = parser.parseSentence(indexedSentence._1)
    val tree = parse.dependencyTree.flatMap(tree => if (shouldKeepTree(tree)) Some(tree) else None)
    (indexedSentence, tree)
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

  def sentenceAndLogicalFormAsString(input: (IndexedSentence, Option[Logic])): String = {
    val sentence = input._1._1
    val index = input._1._2
    val logicalForm = input._2
    val lfString = logicalForm.map(_.toString).mkString(" ")
    index match {
      case None => lfString
      case Some(index) => s"${index}\t${lfString}"
    }
  }
}

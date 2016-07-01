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

import scala.concurrent._
import scala.util.{Success, Failure}
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global

class SentenceToLogic(
  params: JValue,
  fileUtil: FileUtil
) extends Step(Some(params), fileUtil) {
  implicit val formats = DefaultFormats
  override val name = "Sentence to Logic"

  val validParams = Seq("sentences", "logical forms", "output file")
  JsonHelper.ensureNoExtras(params, name, validParams)

  // NOTE: Because of how we decide on the output file name down below, you _must_ make the
  // positive training data file the first sentence input here!  Or you need to manually specify
  // the output file parameter.
  val sentenceInputs = SentenceToLogic.getSentenceInputs(params \ "sentences", fileUtil)
  val sentenceFiles = sentenceInputs.map(_._1)
  val logicalFormGenerator = new LogicalFormGenerator(params \ "logical forms")

  val outputFile = JsonHelper.extractAsOption[String](params, "output file") match {
    case None => {
      sentenceFiles.head.replace("training_data.tsv", "logical_forms.txt")
    }
    case Some(filename) => filename
  }

  val numPartitions = 1

  override val inputs: Set[(String, Option[Step])] = sentenceInputs
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

    parseSentences(sc, logicalFormGenerator, outputFile)

    sc.stop()
  }

  def parseSentences(sc: SparkContext, logicalFormGenerator: LogicalFormGenerator, outputFile: String) {
    fileUtil.mkdirsForFile(outputFile)
    val sentences = sc.textFile(sentenceFiles.mkString(","))
    val trees = sentences.flatMap(SentenceToLogic.parseSentence)
    val logicalForms = trees.flatMap(sentenceAndTree => {
      val result = SentenceToLogic.runWithTimeout(2000, () => {
        val sentence = sentenceAndTree._1
        val tree = sentenceAndTree._2
        val logicalForm = try {
          logicalFormGenerator.getLogicalForm(tree)
        } catch {
          case e: Throwable => { println(sentence); tree.print(); throw e }
        }
        (sentence, logicalForm)
      })
      result match {
        case None => {
          println(s"Timeout while processing sentence: ${sentenceAndTree._1}")
          Seq()
        }
        case Some(either) => either match {
          case Left(t) => {
            println(s"Exception thrown while processing sentence: ${sentenceAndTree._1} ---- ${t.getMessage}")
            Seq()
          }
          case Right(result) => Seq(result)
        }
      }
    })
    val outputStrings = logicalForms.flatMap(sentenceAndLf => {
      val result = SentenceToLogic.runWithTimeout(2000, () => {
        SentenceToLogic.sentenceAndLogicalFormAsString(sentenceAndLf)
      })
      result match {
        case None => {
          println(s"Timeout while printing sentence: ${sentenceAndLf._1}")
          Seq()
        }
        case Some(either) => either match {
          case Left(t) => {
            println(s"Exception thrown while printing sentence: ${sentenceAndLf._1} ---- ${t.getMessage}")
            Seq()
          }
          case Right(result) => result
        }
      }
    })

    val finalOutput = outputStrings.collect()
    fileUtil.writeLinesToFile(outputFile, finalOutput)
  }
}

// This semi-ugliness is so that the spark functions are serializable.
object SentenceToLogic {
  val parser = new StanfordParser

  def getSentenceInputs(params: JValue, fileUtil: FileUtil): Set[(String, Option[Step])] = {
    val sentenceSelector = new SentenceSelectorStep(params \ "sentence selector step", fileUtil)
    Set()
  }

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

  def parseSentence(sentence: String) = {
    val parse = parser.parseSentence(sentence)
    parse.dependencyTree match {
      case None => Seq()
      case Some(tree) => {
        if (shouldKeepTree(tree)) {
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

  def sentenceAndLogicalFormAsString(logicalForm: (String, Option[Logic])): Seq[String] = {
    val sentence = logicalForm._1
    val lf = logicalForm._2
    val lfString = lf.map(_.toString).mkString(" ")
    Seq(s"${sentence}\t${lfString}")
  }
}

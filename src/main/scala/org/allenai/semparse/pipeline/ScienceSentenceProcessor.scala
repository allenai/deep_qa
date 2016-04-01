package org.allenai.semparse.pipeline

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.json4s._

import org.allenai.semparse.parse.DependencyTree
import org.allenai.semparse.parse.LogicalFormGenerator
import org.allenai.semparse.parse.Predicate
import org.allenai.semparse.parse.StanfordParser

class ScienceSentenceProcessor(
  params: JValue,
  fileUtil: FileUtil = new FileUtil
) extends Step(Some(params), fileUtil) {
  override val name = "Science Sentence Processor"

  val validParams = Seq("min word count per sentence", "max word count per sentence",
    "sentences per word file", "data directory", "output file", "output format")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val dataDir = JsonHelper.extractWithDefault(params, "data directory", "/home/mattg/data/petert_science_sentences")
  val outputFile = JsonHelper.extractWithDefault(params, "output file", "data/science_sentences.txt")
  val minWordCount = JsonHelper.extractWithDefault(params, "min word count per sentence", 4)
  val maxWordCount = JsonHelper.extractWithDefault(params, "max word count per sentence", 20)
  val formatOptions = Seq("training data", "debug")
  val outputFormat = JsonHelper.extractOptionWithDefault(params, "output format", formatOptions, "training data")
  val sentencesPerFile = (params \ "sentences per word file") match {
    case JNothing => None
    case JInt(num) => Some(num.toInt)
    case _ => throw new IllegalStateException("'sentences per word file' must be an integer")
  }

  override val inputs: Set[(String, Option[Step])] = Set((dataDir, None))
  override val outputs = Set(outputFile)
  override val paramFile = outputs.head.replace(".txt", "_params.json")

  override def _runStep() {
    fileUtil.mkdirsForFile(outputFile)
    val parser = new StanfordParser
    val files = fileUtil.listDirectoryContents(dataDir)
    for (file <- files.par) {
      val keptSentences = fileUtil.getLineIterator(dataDir + "/" + file).flatMap(line => {
        val sentence = line.replace("<SENT>", "").replace("</SENT>", "")
        if (shouldKeepSentence(sentence)) {
          Seq(sentence)
        } else {
          Seq()
        }
      }).toSet[String].toSeq.sortBy(_.split(" ").length)
      val keptTrees = keptSentences.toIterator.flatMap(sentence => {
        val parse = parser.parseSentence(sentence)
        parse.dependencyTree match {
          case None => Seq()
          case Some(tree) => {
            if (shouldKeepTree(tree)) {
              Seq((sentence, tree))
            } else {
              None
            }
          }
        }
      })
      val logicalForms = keptTrees.map(sentenceAndTree => {
        val sentence = sentenceAndTree._1
        val tree = sentenceAndTree._2
        val logicalForm = try {
          LogicalFormGenerator.getLogicalForm(tree)
        } catch {
          case e: Throwable => { println(sentence); tree.print(); throw e }
        }
        (sentence, logicalForm)
      })
      val finalLogicalForms = sentencesPerFile match {
        case None => logicalForms
        case Some(num) => logicalForms.take(num)
      }
      val outputStrings = finalLogicalForms.toSeq.par.flatMap(logicalFormToString).seq
      fileUtil synchronized {
        fileUtil.writeLinesToFile(outputFile, outputStrings, true)
      }
    }
  }

  def shouldKeepSentence(sentence: String): Boolean = {
    val wordCount = sentence.split(" ").length
    if (wordCount < minWordCount) return false
    if (wordCount > maxWordCount) return false
    if (sentence.endsWith("?") || sentence.endsWith("!")) return false
    if (sentence.contains("(") || sentence.contains(")") || sentence.contains(":")) return false
    if (sentence.contains(">") || sentence.contains("<") || sentence.contains("-")) return false
    if (sentence.contains("\"") || sentence.contains("'")) return false
    if (sentence.contains("&gt;") || sentence.contains("&lt;") || sentence.contains("&quot;")) return false
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

  def logicalFormToString(logicalForm: (String, Set[Predicate])): Seq[String] = {
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

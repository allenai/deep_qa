package org.allenai.semparse.one_off

import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods._

import scala.collection.mutable

import edu.cmu.ml.rtw.users.matt.util.FileUtil

class TestJsonAugmenter {
  implicit val formats = DefaultFormats

  def augmentLine(line: String): String = {
    val json = parse(line)
    compact(render(augmentJson(json)))
  }

  def augmentJson(json: JValue): JValue = {
    val queries = (json \ "queries").extract[List[JValue]]
    val newQueries = queries.map(augmentQuery)
    val newJson: JValue = ("queries" -> newQueries)
    json removeField { _._1 == "queries" } merge newJson
  }

  def augmentQuery(queryJson: JValue): JValue = {
    val expression = (queryJson \ "queryExpression").extract[String]
    val wordRels = getWordRelsFromExpression(expression)
    val wordRelJson: JValue = wordRels.map(entry => {
      List(JString(entry._1), JString(entry._2), JBool(entry._3))
    }).toList
    val extraJson: JValue = ("midRelationsInQuery" -> wordRelJson)
    queryJson removeField { _._1 == "midRelationsInQuery" } merge extraJson
  }

  def getWordRelsFromExpression(expression: String): Seq[(String, String, Boolean)] = {
    val variableName = {
      val parenIndex = expression.indexOf("(", 1)
      val endParenIndex = expression.indexOf(")")
      expression.substring(parenIndex + 1, endParenIndex)
    }
    var index = 0
    var finished = false
    val wordRels = new mutable.ArrayBuffer[(String, String, Boolean)]
    while (!finished) {
      val wordRelIndex = expression.indexOf("((word-rel", index)
      if (wordRelIndex == -1) {
        finished = true
      } else {
        val firstClosingParen = expression.indexOf(")", wordRelIndex)
        val endIndex = expression.indexOf(")", firstClosingParen + 1)
        val wordRelExpression = expression.substring(wordRelIndex + 1, endIndex)
        val parts = wordRelExpression.split(" ")
        val word = parts(1).substring(0, parts(1).length - 1)
        val arg1 = parts(2)
        val arg2 = parts(3)
        if (arg1 == variableName) {
          wordRels += Tuple3(word, arg2, false)
        } else if (arg2 == variableName) {
          wordRels += Tuple3(word, arg1, true)
        }
        index = endIndex
      }
    }
    wordRels.toSeq
  }
}

object add_rel_variables_to_test_file {
  implicit val formats = DefaultFormats
  val testFile = "/home/mattg/clone/tacl2015-factorization/data/acl2016-test-pretty.json"
  val augmentedTestFile = "/home/mattg/clone/tacl2015-factorization/data/acl2016-test-augmented.txt"
  val prettyTestFile = "/home/mattg/clone/tacl2015-factorization/data/acl2016-test-augmented-pretty.json"
  val fileUtil = new FileUtil

  def main(args: Array[String]) {
    val augmenter = new TestJsonAugmenter
    //val newLines = fileUtil.getLineIterator(testFile).map(augmenter.augmentLine).toSeq
    val json = parse(fileUtil.readLinesFromFile(testFile).mkString("\n"))
    val jsonList = json.extract[Seq[JValue]]
    val augmented = jsonList.map(augmenter.augmentJson)
    val newLines = augmented.map(j => compact(render(j)))
    fileUtil.writeLinesToFile(augmentedTestFile, newLines)
    val writer = fileUtil.getFileWriter(prettyTestFile)
    writer.write(pretty(render(augmented)))
    writer.close()
  }
}

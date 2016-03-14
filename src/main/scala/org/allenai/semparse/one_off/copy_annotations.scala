package org.allenai.semparse.one_off

import com.mattg.util.FileUtil

import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods._

object copy_annotations {
  implicit val formats = DefaultFormats
  val fileUtil = new FileUtil
  val copyFrom = "/home/mattg/clone/tacl2015-factorization/src/main/resources/acl2016_dev_set_annotated.json"
  val copyTo = "/home/mattg/clone/tacl2015-factorization/src/main/resources/acl2016_dev_set_old_lfs_annotated.json"

  def copyQueryAnnotations(json: (JValue, JValue)): JValue = {
    val (copyFrom, copyTo) = json
    val correct = copyFrom \ "correctAnswerIds"
    val incorrect = copyFrom \ "incorrectAnswerIds"
    val toCopy: JValue = ("correctAnswerIds" -> correct) ~ ("incorrectAnswerIds" -> incorrect)
    copyTo removeField { _._1 == "incorrectAnswerIds" } removeField { _._1 == "correctAnswerIds" } merge toCopy

  }

  def copySentenceAnnotations(json: (JValue, JValue)): JValue = {
    val (copyFrom, copyTo) = json
    val queriesFrom = (copyFrom \ "queries").extract[Seq[JValue]]
    val queriesTo = (copyTo \ "queries").extract[Seq[JValue]]
    val copiedJson: JValue = queriesFrom.zip(queriesTo).map(copyQueryAnnotations)
    val toCopy: JValue = ("queries" -> copiedJson)
    copyTo removeField { _._1 == "queries" } merge toCopy
  }

  // This script is useful for copying annotations over from the new logical form file to the old
  // logical form file.  Hopefully I won't need to use it again...
  def NOT_main(args: Array[String]) {
    val jsonToCopyFrom = parse(fileUtil.readLinesFromFile(copyFrom).mkString("\n"))
    val queriesToCopyFrom = jsonToCopyFrom.extract[Seq[JValue]]
    val jsonToCopyTo = parse(fileUtil.readLinesFromFile(copyTo).mkString("\n"))
    val queriesToCopyTo = jsonToCopyTo.extract[Seq[JValue]]

    val copiedJson = queriesToCopyFrom.zip(queriesToCopyTo).map(copySentenceAnnotations)

    val writer = fileUtil.getFileWriter(copyTo)
    writer.write(pretty(render(copiedJson)))
    writer.close()
  }
}

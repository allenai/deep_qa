package org.allenai.deep_qa.data

import com.mattg.util.FileUtil
import com.typesafe.scalalogging.LazyLogging

import scala.collection.mutable
import scala.xml
import scala.sys.process.Process

class WhoDidWhatDatasetReader(fileUtil: FileUtil) extends DatasetReader[WhoDidWhatInstance] with LazyLogging {
  override def readFile(filename: String): Dataset[WhoDidWhatInstance] = {
    // We replace non-breaking spaces (&nbsp;) in input files with regular spaces
    // and back up the original file at inputFile.bak
    logger.info(s"""Removing non-breaking spaces in ${filename}, backing up original input file at ${filename}.bak""")
    val command = Seq("sed", "-i.bak", "s/&nbsp;/ /g", filename)
    val process = Process(command)
    val exitCode = process.!
    if (exitCode != 0) {
      throw new RuntimeException("Subprocess returned non-zero exit code: $exitCode")
    }

    val xml = scala.xml.Utility.trim(scala.xml.XML.loadString(fileUtil.readFileContents(filename)))
    val instanceTuples = for {
      mc <- xml \ "mc"

      questionNode = mc \ "question"
      leftContextNode = questionNode \ "leftcontext"
      leftContext = leftContextNode.text
      rightContextNode = questionNode \ "rightcontext"
      rightContext = rightContextNode.text

      passageNode = mc \ "contextart"
      passage = passageNode.text

      answerOptionNodes = mc \ "choice"
      answerOptions = answerOptionNodes.map((answerOption: scala.xml.Node) => answerOption.text)

      labelNodes = answerOptionNodes.filter((answerOption: scala.xml.Node) => (answerOption \ "@correct").text == "true")
      label = (labelNodes \ "@idx").text.toInt
    } yield (passage, leftContext, rightContext, answerOptions, label)

    val instances = instanceTuples.map { case (passage, leftContext, rightContext, answerOptions, label) => {
      WhoDidWhatInstance(passage, leftContext, rightContext, answerOptions, Some(label))
    }}
    Dataset(instances)
  }
}

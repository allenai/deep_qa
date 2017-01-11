package org.allenai.deep_qa.data

import com.mattg.util.FileUtil

import scala.collection.mutable
import scala.xml


class WhoDidWhatDatasetReader(fileUtil: FileUtil) extends DatasetReader[WhoDidWhatInstance] {
  override def readFile(filename: String): Dataset[WhoDidWhatInstance] = {
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

package org.allenai.deep_qa.data

import scala.collection.mutable

import com.mattg.util.FileUtil

class SnliDatasetReader(fileUtil: FileUtil) extends DatasetReader[SnliInstance] {
  override def readFile(filename: String): Dataset[SnliInstance] = {
    val instances = fileUtil.flatMapLinesFromFile(filename, line => {
      val fields = line.split("\t")
      val label = fields(0) match {
        case "neutral" => Some("neutral")
        case "entailment" => Some("entails")
        case "contradiction" => Some("contradicts")
        case _ => None
      }
      val text = fields(5)
      val hypothesis = fields(6)
      label.map(l => SnliInstance(text, hypothesis, Some(l))).toSeq
    })
    Dataset(instances)
  }

  def readFromSentenceAndBackgroundFiles(sentenceFile: String, backgroundFile: String): Dataset[SnliInstance] = {
    val sentences = fileUtil.readLinesFromFile(sentenceFile)
    val background = fileUtil.readLinesFromFile(backgroundFile)
    val instances = sentences.zip(background).map { case (sentence, background) => {
      val sentenceFields = sentence.split("\t")
      val hypothesis = sentenceFields(1)
      val label = if (sentenceFields(2) == "1") "entails" else "contradicts"
      val backgroundFields = background.split("\t")
      val text = backgroundFields(1)
      SnliInstance(text, hypothesis, Some(label))
    }}
    Dataset(instances)
  }
}

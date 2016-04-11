package org.allenai.semparse.pipeline.science_data

import org.json4s._
import org.json4s.JsonDSL._

import com.mattg.util.FileUtil

object ScienceQuestionPipeline {
  val fileUtil = new FileUtil

  val sentenceProcesserParams: JValue =
    ("max word count per sentence" -> 10) ~
    ("data name" -> "petert_sentences") ~
    ("data directory" -> "/home/mattg/data/petert_science_sentences")

  val kbGeneratorParams: JValue = ("sentences" -> sentenceProcesserParams)

  val kbGraphCreatorParams: JValue = ("graph name" -> "petert") ~ ("corpus triples" -> kbGeneratorParams)

  val questionProcesserParams: JValue =
    ("question file" -> "data/monarch_questions/raw_questions.tsv") ~
    ("data name" -> "monarch_questions")

  def main(args: Array[String]) {
    new KbGraphCreator(kbGraphCreatorParams, fileUtil).runPipeline()
    //new ScienceQuestionProcessor(questionProcesserParams, fileUtil).runPipeline()
  }
}

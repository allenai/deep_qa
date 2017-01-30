package org.allenai.deep_qa.experiments

import org.allenai.deep_qa.experiments.datasets.ScienceCorpora
import org.allenai.deep_qa.experiments.datasets.ScienceFiles
import org.allenai.deep_qa.pipeline.SentenceProducer

import com.mattg.util.FileUtil
import org.json4s.JsonDSL._
import org.json4s._

object ExperimentsWithTuples {
  val fileUtil = new FileUtil

  def main(args: Array[String]) {
    val backgroundAsTupleParams: JValue =
      ("sentence producer type" -> "sentence to tuple") ~
      ("sentences" ->
        ("sentence producer type" -> "background searcher") ~
//        ("searcher" -> ScienceCorpora.buscElasticSearchIndex(10)) ~
//        ("sentences" -> ScienceFiles.omnibusGradeFourTrainSentences_multipleTrueFalse_appendAnswer) ~
        ("searcher" -> ScienceCorpora.buscElasticSearchIndex(10)) ~
        ("sentences" -> ScienceFiles.omnibusGradeFourTrainSentences_questionAndAnswer) ~
        ("sentence format" -> "plain sentence")) ~
      ("tuple extractor" ->
        ("type" -> "open ie") ~
        ("max argument characters" -> 50) ~
        ("max sentence characters" -> 300))
    SentenceProducer.create(backgroundAsTupleParams, fileUtil).runPipeline()
  }
}

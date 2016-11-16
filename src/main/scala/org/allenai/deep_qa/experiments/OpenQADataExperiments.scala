package org.allenai.deep_qa.experiments

import org.allenai.deep_qa.experiments.datasets.ScienceDatasets
import org.allenai.deep_qa.pipeline.Evaluator

import com.mattg.util.FileUtil
import org.json4s.JsonDSL._
import org.json4s._

object OpenQADataExperiments {
  val fileUtil = new FileUtil

  val openQaDataMultipleChoiceMemoryNetwork: JValue = {
    val modelParams: JValue =
      Models.multipleTrueFalseMemoryNetwork merge
        Models.basicMemoryNetworkComponents merge
        Debug.multipleTrueFalseDefault merge
        Encoders.bagOfWords merge
        Training.default merge
        //Training.entailmentPretraining merge
        (("max_sentence_length" -> 100) ~ ("patience" -> 20))

    ("model params" -> modelParams) ~
      ("dataset" -> ScienceDatasets.openQaMtfQuestionsWithBuscBackground) ~
      ("validation dataset" -> ScienceDatasets.omnibusMtfGradeFourTrainQuestionsWithBuscBackground)
  }

  val models = Seq(openQaDataMultipleChoiceMemoryNetwork)

  def main(args: Array[String]) {

    new Evaluator(models, fileUtil).runPipeline()
  }
}

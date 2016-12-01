package org.allenai.deep_qa.experiments

import org.allenai.deep_qa.experiments.datasets.CreatedScienceDatasets
import org.allenai.deep_qa.experiments.datasets.ScienceDatasets
import org.allenai.deep_qa.pipeline.Evaluator

import com.mattg.util.FileUtil
import org.json4s.JsonDSL._
import org.json4s._

object MemoryNetworkExperiments {
  val fileUtil = new FileUtil

  def multipleChoiceMemoryNetwork(name: String, trainingDataset: JValue): JValue = {
    val modelParams: JValue =
      Models.multipleTrueFalseMemoryNetwork merge
      Models.betterEntailmentComponents merge
      //Debug.multipleTrueFalseDefault merge
      Encoders.gru merge
      Training.default merge
      //Training.entailmentPretraining merge
      (("max_sentence_length" -> 100) ~ ("patience" -> 20))

    ("name" -> name) ~
    ("model params" -> modelParams) ~
    ("dataset" -> trainingDataset) ~
    ("validation dataset" -> ScienceDatasets.omnibusMtfGradeFourDevQuestionsWithBuscBackground)
  }

  def omnibusWithTruncatedGeneratedQuestions(linesToKeep: Int): JValue = {
    val truncatedQuestions: JValue =
      ("dataset type" -> "truncated") ~
      ("dataset to truncate" -> CreatedScienceDatasets.johannesVersion01WithBuscBackground) ~
      ("instances to keep" -> linesToKeep) ~
      ("output directory" -> s"/efs/data/dlfa/generated_questions/v0.1/truncated_${linesToKeep}/")

    val combinedDataset: JValue =
      ("dataset type" -> "combined") ~
      ("datasets" -> Seq(
        truncatedQuestions,
        ScienceDatasets.omnibusMtfGradeFourTrainQuestionsWithBuscBackground)) ~
      ("output directory" -> s"/efs/data/dlfa/processed/omnibus_4_train_and_${linesToKeep}_generated/")
    combinedDataset
  }

  val omnibusTrain = multipleChoiceMemoryNetwork(
    "omnibus",
    ScienceDatasets.omnibusMtfGradeFourTrainQuestionsWithBuscBackground
  )

  val dataSizes = Seq(500, 1000, 2000, 5000, 10000, 15000, 20000)

  val withGeneratedData = dataSizes.map(numInstances => {
    multipleChoiceMemoryNetwork(
      s"omnibus plus $numInstances generated questions",
      omnibusWithTruncatedGeneratedQuestions(numInstances * 4)
    )
  })

  //val models = Seq(omnibusTrain) ++ withGeneratedData

  def tableMcqExperiment(name: String, modelParams: JValue): JValue = {
    ("name" -> name) ~
    ("model params" -> modelParams) ~
    ("dataset" -> ScienceDatasets.tableMcqTrainWithCorrectBackground) ~
    ("validation dataset" -> ScienceDatasets.tableMcqDevWithCorrectBackground)
  }

  val tableMcqDecomposableAttention = tableMcqExperiment(
    "Decomposable attention",
    Models.decomposableAttentionSolver merge Training.default
  )

  val tableMcqMtfDecomposableAttention = tableMcqExperiment(
    "MultipleTF Decomposable attention",
    Models.multipleTrueFalseDecomposableAttentionSolver merge Training.default
  )

  val tableMcqMtfmnsGru = tableMcqExperiment(
    "MTFMNS with GRU",
    Models.multipleTrueFalseMemoryNetwork merge
    Models.betterEntailmentComponents merge
    //Debug.multipleTrueFalseDefault merge
    Encoders.gru merge
    Training.default
  )

  val tableMcqTfmnsGru = tableMcqExperiment(
    "TFMNS with GRU",
    Models.trueFalseMemoryNetwork merge
    Models.betterEntailmentComponents merge
    //Debug.multipleTrueFalseDefault merge
    Encoders.gru merge
    Training.default
  )

  val models = Seq(
    tableMcqDecomposableAttention,
    tableMcqMtfDecomposableAttention,
    tableMcqMtfmnsGru,
    tableMcqTfmnsGru
  )

  def main(args: Array[String]) {
    new Evaluator(Some("table_mcq_decomposable_attention_2016_12_01"), models, fileUtil).runPipeline()
  }
}

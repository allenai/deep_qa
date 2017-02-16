package org.allenai.deep_qa.experiments

import org.allenai.deep_qa.experiments.datasets.CreatedScienceDatasets
import org.allenai.deep_qa.experiments.datasets.ScienceDatasets
import org.allenai.deep_qa.pipeline.Evaluator

import com.mattg.util.FileUtil
import org.json4s.JsonDSL._
import org.json4s._

object MemoryNetworkExperiments {
  val fileUtil = new FileUtil

  def experiment(
    name: String,
    modelParams: JValue,
    trainingDataset: JValue,
    validationDataset: JValue
  ): JValue = {
    ("name" -> name) ~
    ("model params" -> modelParams) ~
    ("dataset" -> trainingDataset) ~
    ("validation dataset" -> validationDataset)
  }

  def experiment(
    name: String,
    modelParams: JValue,
    trainingDataset: JValue
  ): JValue = {
    ("name" -> name) ~
    ("model params" -> modelParams) ~
    ("dataset" -> trainingDataset)
  }

  def omnibusGradeFourExperiment(
    name: String,
    modelParams: JValue,
    trainingDataset: JValue=ScienceDatasets.omnibusMtfGradeFourTrainQuestionsWithBuscBackground
  ): JValue = {
    experiment(
      name,
      modelParams,
      trainingDataset,
      ScienceDatasets.omnibusMtfGradeFourDevQuestionsWithBuscBackground
    )
  }

  def intermediateExperiment(
    name: String,
    modelParams: JValue,
    trainingDataset: JValue=ScienceDatasets.intermediateMtfGradeFourAndEightTrainQuestionsWithBuscBackground
  ): JValue = {
    experiment(
      name,
      modelParams,
      trainingDataset,
      ScienceDatasets.intermediateMtfGradeFourAndEightDevQuestionsWithBuscBackground
    )
  }

  def tableMcqExperiment(name: String, modelParams: JValue): JValue = {
    experiment(
      name,
      modelParams,
      ScienceDatasets.tableMcqTrainWithCorrectBackground,
      ScienceDatasets.tableMcqDevWithCorrectBackground
    )
  }

  val multipleChoiceMemoryNetwork: JValue = {
    Models.multipleTrueFalseMemoryNetwork merge
    Models.betterEntailmentComponents merge
    //Debug.multipleTrueFalseDefault merge
    Encoders.gru merge
    Training.default merge
    //Training.entailmentPretraining merge
    (("max_sentence_length" -> 100) ~ ("patience" -> 20))
  }

  def omnibusWithTruncatedGeneratedQuestions(version: String, linesToKeep: Int): JValue = {
    val truncatedQuestions: JValue =
      ("dataset type" -> "truncated") ~
      ("dataset to truncate" -> CreatedScienceDatasets.johannesBackgroundDataset(version)) ~
      ("instances to keep" -> linesToKeep) ~
      ("output directory" -> s"/efs/data/dlfa/generated_questions/v${version}/truncated_${linesToKeep}/")

    val combinedDataset: JValue =
      ("dataset type" -> "combined") ~
      ("datasets" -> Seq(
        truncatedQuestions,
        ScienceDatasets.omnibusMtfGradeFourTrainQuestionsWithBuscBackground)) ~
      ("output directory" -> s"/efs/data/dlfa/processed/omnibus_4_train_and_${linesToKeep}_generated_v${version}/")
    combinedDataset
  }

  def main(args: Array[String]) {
    //runExperimentWithTurkedQuestions()
    //runGeneratedQuestionsExperiment()
    runIntermediateMemoryNetworkExperiment()
  }

  def runIntermediateMemoryNetworkExperiment() {
    val baseline = intermediateExperiment(
      "baseline mtfmn",
      multipleChoiceMemoryNetwork
    )
    val models = Seq(baseline)
    new Evaluator(Some("baseline_mtfmn_on_intermediate_set_2017_02_15"), models, fileUtil).runPipeline()
  }

  def runExperimentWithTurkedQuestions() {
    val omnibusTrain = omnibusGradeFourExperiment(
      "omnibus",
      multipleChoiceMemoryNetwork,
      ScienceDatasets.omnibusMtfGradeFourTrainQuestionsWithBuscBackground
    )

    val dataSizes = Seq(250, 500, 750, 1000, 1250, 1500)

    val withGeneratedData = dataSizes.map(numInstances => {
      omnibusGradeFourExperiment(
        s"omnibus plus $numInstances turked questions",
        multipleChoiceMemoryNetwork,
        omnibusWithTruncatedGeneratedQuestions("0.5", numInstances * 4)
      )
    })

    val models = Seq(omnibusTrain) ++ withGeneratedData
    new Evaluator(Some("turked_questions_plus_omnibus_2016_12_05"), models, fileUtil).runPipeline()
  }

  def runGeneratedQuestionsExperiment() {
    val omnibusTrain = omnibusGradeFourExperiment(
      "omnibus",
      multipleChoiceMemoryNetwork,
      ScienceDatasets.omnibusMtfGradeFourTrainQuestionsWithBuscBackground
    )

    val dataSizes = Seq(500, 1000, 2000, 5000, 10000, 15000, 20000)

    val withGeneratedData = dataSizes.map(numInstances => {
      omnibusGradeFourExperiment(
        s"omnibus plus $numInstances generated questions",
        multipleChoiceMemoryNetwork,
        omnibusWithTruncatedGeneratedQuestions("0.1", numInstances * 4)
      )
    })

    val models = Seq(omnibusTrain) ++ withGeneratedData
    new Evaluator(Some("generated_questions_plus_omnibus_2016_12_05"), models, fileUtil).runPipeline()
  }

  def runDecomposableAttentionExperimentOnOmnibus() {
    val modelParams: JValue =
      Models.multipleTrueFalseDecomposableAttentionSolver merge
      Training.default

    val experiment: JValue =
      ("name" -> "omnibus decomposable attention") ~
      ("model params" -> modelParams) ~
      ("dataset" -> ScienceDatasets.omnibusMtfGradeFourTrainQuestionsWithBuscBackground) ~
      ("validation dataset" -> ScienceDatasets.omnibusMtfGradeFourDevQuestionsWithBuscBackground)

    val models = Seq(
      omnibusGradeFourExperiment(
        "omnibus decomposable attention",
        Models.multipleTrueFalseDecomposableAttentionSolver merge Training.default
      ),
      omnibusGradeFourExperiment(
        "omnibus decomposable attention with pretrained embeddings (with projection, no fine tuning)",
        Models.multipleTrueFalseDecomposableAttentionSolver merge Training.default merge
        Encoders.pretrainedGloveEmbeddings(false, true)
      ),
      omnibusGradeFourExperiment(
        "omnibus decomposable attention with pretrained embeddings (no projection, no fine tuning)",
        Models.multipleTrueFalseDecomposableAttentionSolver merge Training.default merge
        Encoders.pretrainedGloveEmbeddings(false, false)
      ),
      omnibusGradeFourExperiment(
        "omnibus decomposable attention with pretrained embeddings (no projection, with fine tuning)",
        Models.multipleTrueFalseDecomposableAttentionSolver merge Training.default merge
        Encoders.pretrainedGloveEmbeddings(true, false)
      )
    )
    new Evaluator(Some("omnibus_decomposable_attention_2016_12_01"), models, fileUtil).runPipeline()
  }

  def runTableMcqExperiments() {
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
    new Evaluator(Some("table_mcq_decomposable_attention_2016_12_01"), models, fileUtil).runPipeline()
  }
}

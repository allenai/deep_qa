package org.allenai.deep_qa.experiments

import org.allenai.deep_qa.experiments.datasets.SciQDatasets
import org.allenai.deep_qa.experiments.datasets.ScienceDatasets
import org.allenai.deep_qa.pipeline.Evaluator

import com.mattg.util.FileUtil
import org.json4s.JsonDSL._
import org.json4s._

object SciQDatasetExperiments {
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

  /*
   * Create combined datasets with Omnibus4Train + SciQ Train and
   * Omnibus8Train + SciQ Train.
   */

  val combinedSciQTrainOmnibusEightTrainDataset: JValue =
    ("dataset type" -> "combined") ~
      ("datasets" -> Seq(SciQDatasets.sciQTrainDataset, ScienceDatasets.omnibusEightTrainReadingComprehensionDataset))~
      ("output directory" -> s"/efs/data/dlfa/processed/omnibus_8_train_and_sciq_train_combined/")

  val combinedSciQTrainOmnibusFourTrainDataset: JValue =
    ("dataset type" -> "combined") ~
      ("datasets" -> Seq(SciQDatasets.sciQTrainDataset, ScienceDatasets.omnibusFourTrainReadingComprehensionDataset))~
      ("output directory" -> s"/efs/data/dlfa/processed/omnibus_4_train_and_sciq_train_combined/")

  def omnibusGradeFourExperiment(
    name: String,
    modelParams: JValue,
    trainingDataset: JValue=ScienceDatasets.omnibusFourTrainReadingComprehensionDataset
  ): JValue = {
    experiment(
      name,
      modelParams,
      trainingDataset,
      ScienceDatasets.omnibusFourDevReadingComprehensionDataset
    )
  }

  def omnibusGradeEightExperiment(
    name: String,
    modelParams: JValue,
    trainingDataset: JValue=ScienceDatasets.omnibusEightTrainReadingComprehensionDataset
  ): JValue = {
    experiment(
      name,
      modelParams,
      trainingDataset,
      ScienceDatasets.omnibusEightDevReadingComprehensionDataset
    )
  }

  val attentionSumReader: JValue = {
    Models.attentionSumReader
  }

  val gatedAttentionReader: JValue = {
    Models.gatedAttentionReader
  }


  def main(args: Array[String]) {
    runASReaderOmnibusEightExperiment()
    runASReaderOmnibusFourExperiment()
    runGAReaderOmnibusEightExperiment()
    runGAReaderOmnibusFourExperiment()
  }

  def runASReaderOmnibusFourExperiment() {
    val asReaderOmnibusFourDefault = omnibusGradeFourExperiment(
      "ASReader omnibus four",
      attentionSumReader,
      ScienceDatasets.omnibusFourTrainReadingComprehensionDataset
    )

    val asReaderOmnibusFourWithSciQDataset = omnibusGradeFourExperiment(
      "ASReader omnibus four plus SciQ Dataset",
      attentionSumReader,
      combinedSciQTrainOmnibusFourTrainDataset
    )

    val models = Seq(asReaderOmnibusFourDefault, asReaderOmnibusFourWithSciQDataset)
    new Evaluator(Some("ASReader_omnibus_four_plus_sciq_dataset"), models, fileUtil).runPipeline()
  }

  def runASReaderOmnibusEightExperiment() {
    val asReaderOmnibusEightDefault = omnibusGradeEightExperiment(
      "ASReader omnibus eight",
      attentionSumReader,
      ScienceDatasets.omnibusEightTrainReadingComprehensionDataset
    )

    val asReaderOmnibusEightWithSciQDataset = omnibusGradeEightExperiment(
      "ASReader omnibus eight plus SciQ Dataset",
      attentionSumReader,
      combinedSciQTrainOmnibusEightTrainDataset
    )

    val models = Seq(asReaderOmnibusEightDefault, asReaderOmnibusEightWithSciQDataset)
    new Evaluator(Some("ASReader_omnibus_eight_plus_sciq_dataset"), models, fileUtil).runPipeline()
  }

    def runGAReaderOmnibusFourExperiment() {
    val gaReaderOmnibusFourDefault = omnibusGradeFourExperiment(
      "GAReader omnibus four",
      gatedAttentionReader,
      ScienceDatasets.omnibusFourTrainReadingComprehensionDataset
    )

    val gaReaderOmnibusFourWithSciQDataset = omnibusGradeFourExperiment(
      "GAReader omnibus four plus SciQ Dataset",
      gatedAttentionReader,
      combinedSciQTrainOmnibusFourTrainDataset
    )

    val models = Seq(gaReaderOmnibusFourDefault, gaReaderOmnibusFourWithSciQDataset)
    new Evaluator(Some("GAReader_omnibus_four_plus_sciq_dataset"), models, fileUtil).runPipeline()
  }

  def runGAReaderOmnibusEightExperiment() {
    val gaReaderOmnibusEightDefault = omnibusGradeEightExperiment(
      "GAReader omnibus eight",
      gatedAttentionReader,
      ScienceDatasets.omnibusEightTrainReadingComprehensionDataset
    )

    val gaReaderOmnibusEightWithSciQDataset = omnibusGradeEightExperiment(
      "GAReader omnibus eight plus SciQ Dataset",
      gatedAttentionReader,
      combinedSciQTrainOmnibusEightTrainDataset
    )

    val models = Seq(gaReaderOmnibusEightDefault, gaReaderOmnibusEightWithSciQDataset)
    new Evaluator(Some("GAReader_omnibus_eight_plus_sciq_dataset"), models, fileUtil).runPipeline()
  }

}

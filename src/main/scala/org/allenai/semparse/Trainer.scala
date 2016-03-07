package org.allenai.semparse

import java.io.File

import edu.cmu.ml.rtw.users.matt.util.FileUtil

object Trainer {

  def canTrainConfig(data: String, modelType: String, ranking: String, ensembledEvaluation: Boolean) = {
    // At this point, we just need to filter out the baseline model, and we can train everything
    // else.
    modelType match {
      case "baseline" => false
      case _ => true
    }
  }

  def getModelFile(data: String, ranking: String, modelType: String) = {
    val model = modelType match {
      case "baseline" => modelType
      case other => s"${modelType}/${ranking}"
    }
    val ending = modelType match { case "baseline" => ".lisp"; case other => ".ser" }
    s"output/${data}/${model}/model${ending}"
  }

  def createTrainingEnvironment(
    data: String,
    modelType: String,
    ranking: String,
    ensembledEvaluation: Boolean
  ): Environment = {
    val dataFiles = data match {
      case "large" => ranking match {
        case "predicate" => Experiments.PREDICATE_RANKING_LARGE_DATA
        case "query" => Experiments.QUERY_RANKING_LARGE_DATA
      }
      case "small" => ranking match {
        case "predicate" => Experiments.PREDICATE_RANKING_SMALL_DATA
        case "query" => Experiments.QUERY_RANKING_SMALL_DATA
      }
      case other => throw new RuntimeException("unrecognized data option")
    }

    val modelFile = modelType match {
      case "baseline" => throw new IllegalStateException("we don't handle training the baseline")
      case "formal" =>  Experiments.FORMAL_MODEL_FILE
      case "distributional" => Experiments.DISTRIBUTIONAL_MODEL_FILE
      case "combined" => Experiments.COMBINED_MODEL_FILE
      case other => throw new RuntimeException("unrecognized model type")
    }

    val rankingFile = ranking match {
      case "predicate" => Experiments.PREDICATE_RANKING_FILE
      case "query" => Experiments.QUERY_RANKING_FILE
      case other => throw new RuntimeException("unrecognized ranking type")
    }

    val sfeSpecFile = data match {
      case "large" => Experiments.LARGE_SFE_SPEC_FILE
      case "small" => Experiments.SMALL_SFE_SPEC_FILE
      case other => throw new RuntimeException("unrecognized data option")
    }

    val serializedModelFile = getModelFile(data, ranking, modelType)

    val inputFiles =
      dataFiles ++ Experiments.ENV_FILES ++ Seq(rankingFile, modelFile, Experiments.TRAIN_FILE)
    val extraArgs = Seq(sfeSpecFile, serializedModelFile)


    new Environment(inputFiles, extraArgs, true)
  }

  def main(args: Array[String]) {
    val fileUtil = new FileUtil

    // We train a model for each experiment listed in Experiments.experimentConfigs, if the
    // configuration is one that we can actually train, and if the model file isn't already
    // present.
    //
    // And, for now, we'll do this sequentially, as parallel output in the terminal would be a big
    // mess (not to mention it'd probably take too much memory).
    Experiments.experimentConfigs.foreach(config => {
      val (data, modelType, ranking, usingGraphs) = config
      val modelFile = getModelFile(data, ranking, modelType)
      if (!canTrainConfig(data, modelType, ranking, usingGraphs)) {
        println(s"Configuration $config is not trainable.  Skipping...")
      } else if (fileUtil.fileExists(modelFile)) {
        println(s"Model already trained configuration $config (model file: $modelFile).  Skipping...")
      } else {
        println(s"Training model for $config")
        fileUtil.mkdirs(new File(modelFile).getParent())
        println("Creating environment (which trains and outputs the model)")
        val env = createTrainingEnvironment(data, modelType, ranking, usingGraphs)
      }
    })
  }
}

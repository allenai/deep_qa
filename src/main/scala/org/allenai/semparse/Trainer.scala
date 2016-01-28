package org.allenai.semparse

class Trainer {
}

object Trainer {
  def getModelFile(data: String, ranking: String, usingGraphs: Boolean, baseline: Boolean) = {
    val graphs = if (usingGraphs) "with_graph_features" else "no_graph_features"
    baseline match {
      case true => s"output/${data}/baseline/model.lisp"
      case false => s"output/${data}/${graphs}/${ranking}/model.ser"
    }
  }

  def createTrainingEnvironment(
    data: String,
    modelType: String,
    ranking: String,
    usingGraphs: Boolean
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
      case "ensemble" => throw new IllegalStateException("you don't train an ensemble...")
      case "uschema" => usingGraphs match {
        case true => Experiments.GRAPH_MODEL_FILE
        case false => Experiments.DISTRIBUTIONAL_MODEL_FILE
      }
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

    val serializedModelFile = getModelFile(data, ranking, usingGraphs, modelType == "baseline")

    val inputFiles =
      dataFiles ++ Experiments.ENV_FILES ++ Seq(rankingFile, modelFile, Experiments.TRAIN_FILE)
    val extraArgs = Seq(sfeSpecFile, serializedModelFile)


    new Environment(inputFiles, extraArgs, true)
  }
}

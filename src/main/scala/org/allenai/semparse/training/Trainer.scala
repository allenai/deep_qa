package org.allenai.semparse.training

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
}

package org.allenai.semparse.one_off

import scala.collection.JavaConverters._

import org.allenai.semparse.Environment
import org.allenai.semparse.Experiments
import org.allenai.semparse.Trainer

import com.jayantkrish.jklol.lisp.ListParameterSpec
import com.jayantkrish.jklol.lisp.SpecAndParameters
import com.jayantkrish.jklol.models.parametric.TensorSufficientStatistics
import com.jayantkrish.jklol.models.TableFactor
import com.jayantkrish.jklol.tensor.SparseTensor
import com.jayantkrish.jklol.util.IndexedList

import edu.cmu.ml.rtw.users.matt.util.FileUtil

object dump_feature_weights {

  val fileUtil = new FileUtil

  def getCatWordParamIndex(modelType: String): Int = modelType match {
    case "formal" => 0
    case "combined" => 1
    case "distributional" => throw new IllegalStateException("distributional model has no such parameters")
  }

  def getRelWordParamIndex(modelType: String): Int = modelType match {
    case "formal" => 1
    case "combined" => 4
    case "distributional" => throw new IllegalStateException("distributional model has no such parameters")
  }

  def getWordParams(params: SpecAndParameters, index: Int): SpecAndParameters = {
      val paramsList = params.getParameterSpec.asInstanceOf[ListParameterSpec]
      val childSpec = paramsList.get(index)
      val childParams = paramsList.getParameter(index, params.getParameters())
      new SpecAndParameters(childSpec, childParams)
  }

  def writeParams(filename: String, words: IndexedList[Object], params: SpecAndParameters) {
    val out = fileUtil.getFileWriter(filename)
    for ((word, index) <- words.items.asScala.map(_.asInstanceOf[String]).zipWithIndex) {
      out.write(word)
      out.write("\n")
      val wordParams = getWordParams(params, index)
      val tensor = wordParams.getParameters().asInstanceOf[TensorSufficientStatistics]
      val summed = tensor.get().elementwiseProduct(SparseTensor.vector(1, 2, Array(1.0, -1.0))).sumOutDimensions(1)
      val numMap = tensor.getStatisticNames().intersection(0)
      val factor = new TableFactor(numMap, summed)
      val assignments = factor.getMostLikelyAssignments(numMap.getDiscreteVariables.get(0).numValues)
      out.write(factor.describeAssignments(assignments))
      out.write("\n")
    }
    out.close()
  }

  def main(args: Array[String]) {
    Experiments.experimentConfigs.foreach(config => {
      val (data, modelType, ranking, ensembledEvaluation) = config
      val wordFile = s"data/${data}/just_words.lisp"
      val modelFile = Trainer.getModelFile(data, ranking, modelType)
      val catWordParamIndex = getCatWordParamIndex(modelType)
      val relWordParamIndex = getRelWordParamIndex(modelType)
      println(s"Loading initial environment (word file: $wordFile)")
      val env = new Environment(Seq(wordFile), Seq())
      println("Getting cat and rel words")
      val catWords = env.evaluateSExpression("cat-words").getValue().asInstanceOf[IndexedList[Object]]
      val relWords = env.evaluateSExpression("rel-words").getValue().asInstanceOf[IndexedList[Object]]
      println(s"Deserializing the model from $modelFile")
      val params = env.evaluateSExpression("(deserialize \""+ modelFile + "\")").getValue().asInstanceOf[SpecAndParameters]

      val paramsList = params.getParameterSpec.asInstanceOf[ListParameterSpec]
      val catWordParams = getWordParams(params, catWordParamIndex)
      val relWordParams = getWordParams(params, relWordParamIndex)
      println("Outputting cat word features")
      val catWordFile = modelFile.replace("model.ser", "cat_word_feature_weights.txt")
      writeParams(catWordFile, catWords, catWordParams)
      println("Outputting rel word features")
      val relWordFile = modelFile.replace("model.ser", "rel_word_feature_weights.txt")
      writeParams(relWordFile, relWords, relWordParams)
    })
  }
}

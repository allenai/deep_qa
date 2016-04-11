package org.allenai.semparse.pipeline.base

import scala.collection.JavaConverters._

import org.allenai.semparse.Environment

import com.jayantkrish.jklol.lisp.ListParameterSpec
import com.jayantkrish.jklol.lisp.SpecAndParameters
import com.jayantkrish.jklol.models.parametric.TensorSufficientStatistics
import com.jayantkrish.jklol.models.TableFactor
import com.jayantkrish.jklol.tensor.SparseTensor
import com.jayantkrish.jklol.util.IndexedList

import com.mattg.util.FileUtil
import com.mattg.pipeline.Step

import org.json4s._

class FeatureWeightDumper(
  params: JValue,
  fileUtil: FileUtil
) extends Step(None, fileUtil) {

  val trainer = new Trainer(params, fileUtil)

  val modelFile = trainer.serializedModelFile

  val wordFile = s"data/${trainer.dataName}/just_words.lisp"
  val catWordFile = modelFile.replace("model.ser", "cat_word_feature_weights.txt")
  val relWordFile = modelFile.replace("model.ser", "rel_word_feature_weights.txt")

  override def inputs = Set((wordFile, None), (modelFile, Some(trainer)))
  override val outputs = Set(catWordFile, relWordFile)
  override val inProgressFile = modelFile.replace("model.ser", "dumper_in_progress")

  val catWordParamIndex = trainer.modelType match {
    case "formal" => 0
    case "combined" => 1
    case "distributional" => throw new IllegalStateException("distributional model has no such parameters")
  }

  val relWordParamIndex = trainer.modelType match {
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

  override def _runStep() {
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
    writeParams(catWordFile, catWords, catWordParams)
    println("Outputting rel word features")
    writeParams(relWordFile, relWords, relWordParams)
  }
}

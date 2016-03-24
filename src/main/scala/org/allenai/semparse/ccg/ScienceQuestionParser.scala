package org.allenai.semparse.ccg

import scala.collection.JavaConverters._

import java.io.File
import java.util.concurrent.TimeUnit
import com.google.common.base.Stopwatch

import edu.uw.easysrl.dependencies.Coindexation
import edu.uw.easysrl.main.EasySRL.InputFormat
import edu.uw.easysrl.main.InputReader
import edu.uw.easysrl.main.ParsePrinter
import edu.uw.easysrl.semantics._
import edu.uw.easysrl.semantics.Logic.LogicVisitor
import edu.uw.easysrl.syntax.grammar.Category
import edu.uw.easysrl.syntax.model.CutoffsDictionary
import edu.uw.easysrl.syntax.model.SRLFactoredModel.SRLFactoredModelFactory
import edu.uw.easysrl.syntax.model.SupertagFactoredModel.SupertagFactoredModelFactory
import edu.uw.easysrl.syntax.model.feature.Feature.FeatureKey
import edu.uw.easysrl.syntax.model.feature.FeatureSet
import edu.uw.easysrl.syntax.parser.ParserAStar
import edu.uw.easysrl.syntax.parser.SRLParser.BackoffSRLParser
import edu.uw.easysrl.syntax.parser.SRLParser.JointSRLParser
import edu.uw.easysrl.syntax.parser.SRLParser.PipelineSRLParser
import edu.uw.easysrl.syntax.parser.SRLParser.SemanticParser
import edu.uw.easysrl.syntax.tagger.POSTagger
import edu.uw.easysrl.syntax.tagger.Tagger
import edu.uw.easysrl.syntax.tagger.TaggerEmbeddings
import edu.uw.easysrl.syntax.training.PipelineTrainer.LabelClassifier
import edu.uw.easysrl.util.Util

import com.mattg.util.JsonHelper

import org.json4s.JValue

class ScienceQuestionParser(params: JValue) {

  // Here we have configuration stuff for the EasySRL parser.  Some of these parameters are
  // obvious, but others aren't as much.  See the documentation to EasySRL for a better description
  // of what these are for.

  // TODO(matt): it might make sense to just have both models (lazily) available at run time,
  // because in parsing questions, sometimes it's an actual question, and sometimes it's a
  // fill-in-the-blank.
  val useQuestionModel = JsonHelper.extractWithDefault(params, "use question model", false)
  val maxLength = JsonHelper.extractWithDefault(params, "max sentence length", 70)
  val nBest = JsonHelper.extractWithDefault(params, "n best", 1)
  val rootCategories = JsonHelper.extractWithDefault(
    params,
    "root categories",
    Seq("S[dcl]", "S[wq]", "S[q]", "S[b]\\NP", "NP")
  ).map(Category.valueOf).asJava
  val superTaggerBeam = JsonHelper.extractWithDefault(params, "supertagger beam", 0.01)
  val superTaggerWeight = JsonHelper.extractWithDefault(params, "supertagger weight", 1.0)
  val questionModelDir = JsonHelper.extractWithDefault(params, "question model dir", "models/easysrl_questions/")
  val sentenceModelDir = JsonHelper.extractWithDefault(params, "sentence model dir", "models/easysrl_sentences/")

  // A few non-parameter fields we'll need below.
  val modelDir = if (useQuestionModel) new File(questionModelDir) else new File(sentenceModelDir)
  val pipelineDir = new File(modelDir, "/pipeline")
  val lexiconFile = new File(modelDir, "lexicon")
  val reader = InputReader.make(InputFormat.TOKENIZED)

  // And these are the big things that take a while to load.  They are all loaded lazily.  This
  // particular setup came from an attempt to greatly simplify what I saw in main.EasySRL in Mike
  // Lewis' repository.  I don't know much about _why_ it's set up this way, just that this is a
  // simplified version of how Mike did it.
  lazy val posTagger = {
    println("Loading POS tagger")
    POSTagger.getStanfordTagger(new File(pipelineDir, "posTagger"))
  }
  lazy val pipeline = {
    println("Making pipeline parser")
    val supertaggerBeam = 0.000001
    val labelClassifier = new File(pipelineDir, "labelClassifier")
    val classifier = Util.deserialize[LabelClassifier](labelClassifier)

    val lexicalCategories = TaggerEmbeddings.loadCategories(new File(modelDir, "categories"))

    // ACK!  Did he really just save global state in this static Coindexation object?  Bleh...
    Coindexation.parseMarkedUpFile(new File(pipelineDir, "markedup"))
    val modelFactory = new SupertagFactoredModelFactory(Tagger.make(pipelineDir, supertaggerBeam, 50, null), lexicalCategories, nBest > 1)
    val aStarParser = new ParserAStar(modelFactory, maxLength, nBest, rootCategories, pipelineDir, 100000)

    new PipelineSRLParser(aStarParser, classifier, posTagger)
  }
  lazy val backoffParser = {
    println("Making backoff parser")
    Coindexation.parseMarkedUpFile(new File(modelDir, "markedup"))
    val cutoffsFile = new File(modelDir, "cutoffs")
    val cutoffs = Util.deserialize[CutoffsDictionary](cutoffsFile)

    val keyToIndex = Util.deserialize[java.util.Map[FeatureKey, Integer]](new File(modelDir, "featureToIndex"))
    val weights = Util.deserialize[Array[Double]](new File(modelDir, "weights"))
    weights(0) = superTaggerWeight
    val featuresFile = new File(modelDir, "features")
    val featureSet = Util.deserialize[FeatureSet](featuresFile).setSupertaggingFeature(pipelineDir, superTaggerBeam)
    val taggerEmbeddings = TaggerEmbeddings.loadCategories(new File(modelDir, "categories"))
    val modelFactory = new SRLFactoredModelFactory(weights, featureSet, taggerEmbeddings, cutoffs, keyToIndex)

    val aStarParser = new ParserAStar(modelFactory, maxLength, nBest, rootCategories, modelDir, 20000)
    new BackoffSRLParser(new JointSRLParser(aStarParser, posTagger), pipeline)
  }
  lazy val lexicon = ScienceQuestionLexicon.getDefault(lexiconFile)
  lazy val parser = {
    println("Making final parser")
    new SemanticParser(backoffParser, lexicon)
  }

  // Done with the set up.  Now we can actually parse things.

  def parseSentences(sentences: Seq[String]): Seq[Seq[Logic]] = {
    val timer = Stopwatch.createStarted()
    val result = sentences.par.map(parseSentence).seq
    val sentencesPerSecond = 1000.0 * sentences.size / timer.elapsed(TimeUnit.MILLISECONDS)
    System.err.println(s"Sentences parsed: ${sentences.size}")
    System.err.println(f"Speed: $sentencesPerSecond%.2f sentences per second")
    result
  }

  def parseSentence(sentence: String): Seq[Logic] = {
    val parses = parser.parseTokens(reader.readInput(sentence).getInputWords()).asScala
    for (parse <- parses) { println(parse.getCcgParse) }
    parses.map(_.getCcgParse.getSemantics.get)
  }
}

class DebugPrintVisitor extends LogicVisitor {
  var depth = 0

  def printIndent() {
    for (i <- 0 until depth) {
      print("  ")
    }
  }

  override def visit(l: AtomicSentence) {
    printIndent()
    println(s"Atomic sentence: $l")
    depth += 1
    for (child <- l.getChildren.asScala) {
      child.accept(this)
    }
    depth -= 1
  }

  override def visit(l: ConnectiveSentence) {
    printIndent()
    println(s"Connective sentence: $l")
    depth += 1
    for (child <- l.getChildren.asScala) {
      child.accept(this)
    }
    depth -= 1
  }

  override def visit(l: Constant) {
    printIndent()
    println(s"Constant: $l")
  }

  override def visit(l: QuantifierSentence) {
    printIndent()
    println(s"Quantifier sentence: $l")
    depth += 1
    l.getChild.accept(this)
    depth -= 1
  }

  override def visit(l: OperatorSentence) {
    printIndent()
    println(s"Operator sentence: $l")
    depth += 1
    l.getScope.accept(this)
    depth -= 1
  }

  override def visit(l: Set) {
    printIndent()
    println(s"Set: $l")
    depth += 1
    for (child <- l.getChildren.asScala) {
      child.accept(this)
    }
    depth -= 1
  }

  override def visit(l: SkolemTerm) {
    printIndent()
    println(s"Skolem term: $l")
    depth += 1
    l.getCondition.accept(this)
    depth -= 1
  }

  override def visit(l: Variable) {
    printIndent()
    println(s"Variable: $l")
  }

  override def visit(l: LambdaExpression) {
    printIndent()
    println(s"Lambda expression: $l")
    depth += 1
    l.getStatement.accept(this)
    depth -= 1
  }

  override def visit(l: Function) {
    printIndent()
    println(s"Function: $l")
    depth += 1
    for (child <- l.getChildren.asScala) {
      child.accept(this)
    }
    depth -= 1
  }
}

object test_easysrl {
  def main(args: Array[String]) {
    import org.json4s.JNothing
    import org.json4s.JsonDSL._
    val parser = new ScienceQuestionParser(("use question model" -> false))
    val questions = Seq(
      //"Which type of energy does a person use to pedal a bicycle?",
      //"Which object is the best conductor of electricty?",
      //"Which characteristic can a human offspring inherit?",
      "Cells contain genetic material called DNA.",
      "Most of Earth is covered by water.",
      "Humans depend on plants for oxygen.",
      "The seeds of an oak come from the fruit.",
      "Which gas is given off by plants?",
      "Matter that is vibrating is producing sound.",
      "Which of these is an example of liquid water?",
      "A human offspring can inherit blue eyes."
    )
    for (question <- questions) {
      val logic = parser.parseSentence(question)(0)
      println(question)
      println(logic)
      val v = new DebugPrintVisitor()
      logic.accept(v)
    }
  }
}

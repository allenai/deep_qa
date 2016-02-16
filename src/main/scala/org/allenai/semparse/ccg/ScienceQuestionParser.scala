package org.allenai.semparse.ccg

import scala.collection.JavaConverters._

import java.io.File

import java.io.BufferedWriter
import java.io.File
import java.io.OutputStreamWriter
import java.text.DecimalFormat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

import com.google.common.base.Stopwatch

import edu.uw.easysrl.dependencies.Coindexation
import edu.uw.easysrl.main.EasySRL.InputFormat
import edu.uw.easysrl.main.InputReader
import edu.uw.easysrl.main.ParsePrinter
import edu.uw.easysrl.semantics.lexicon.CompositeLexicon
import edu.uw.easysrl.syntax.evaluation.CCGBankEvaluation
import edu.uw.easysrl.syntax.grammar.Category
import edu.uw.easysrl.syntax.model.SRLFactoredModel.SRLFactoredModelFactory
import edu.uw.easysrl.syntax.model.SupertagFactoredModel.SupertagFactoredModelFactory
import edu.uw.easysrl.syntax.model.feature.Feature.FeatureKey
import edu.uw.easysrl.syntax.model.feature.FeatureSet
import edu.uw.easysrl.syntax.parser.Parser
import edu.uw.easysrl.syntax.parser.Parser
import edu.uw.easysrl.syntax.parser.ParserAStar
import edu.uw.easysrl.syntax.parser.SRLParser.BackoffSRLParser
import edu.uw.easysrl.syntax.parser.SRLParser.JointSRLParser
import edu.uw.easysrl.syntax.parser.SRLParser.PipelineSRLParser
import edu.uw.easysrl.syntax.parser.SRLParser.SemanticParser
import edu.uw.easysrl.syntax.tagger.POSTagger
import edu.uw.easysrl.syntax.tagger.Tagger
import edu.uw.easysrl.syntax.tagger.TaggerEmbeddings
import edu.uw.easysrl.util.Util

class ScienceQuestionParser(useQuestionModel: Boolean) {

  val numThreads = 1
  val inputFormat = "tokenized"
  val outputFormat = "logic"
  val parsingAlgorithm = "astar"
  val defaultMaxLength = 70
  val defaultNBest = 1
  val rootCategories = Seq("S[dcl]", "S[wq]", "S[q]", "S[b]\\NP", "NP")
  val defaultSuperTaggerBeam = 0.01
  val defaultSuperTaggerWeight = 1.0
  val printer = ParsePrinter.LOGIC_PRINTER

  val questionModelDir = "models/easysrl_questions/"
  val sentenceModelDir = "models/easysrl_sentences/"
  val modelDir = if (useQuestionModel) questionModelDir else sentenceModelDir
  val pipelineDir = new File(modelDir, "/pipeline")

  def main(args: Array[String]) {
    System.err.println("====Starting loading model====")

    val parser2 = if (pipelineDir.exists()) {
      val posTagger = POSTagger.getStanfordTagger(new File(pipelineDir, "posTagger"))
      val pipeline = makePipelineParser(0.000001, true)
      new BackoffSRLParser(new JointSRLParser(makeParser(defaultSuperTaggerBeam, 20000, true, Some(defaultSuperTaggerWeight), defaultNBest, defaultMaxLength), posTagger), pipeline)
    } else {
      makePipelineParser(0.000001, true)
    }

    val parser = {
      val lexiconFile = new File(modelDir, "lexicon")
      val lexicon = if(lexiconFile.exists()) CompositeLexicon.makeDefault(lexiconFile) else CompositeLexicon.makeDefault()
      new SemanticParser(parser2, lexicon)
    }

    val reader = InputReader.make(InputFormat.TOKENIZED)

    val input = Seq("turtles eat fish")
    System.err.println("===Model loaded: parsing...===")

    val timer = Stopwatch.createStarted()
    val parsedSentences = new AtomicInteger()
    val executorService = Executors.newFixedThreadPool(numThreads)

    val sysout = new BufferedWriter(new OutputStreamWriter(System.out))

    var id = 0
    for (line <- input) {
      if (!line.isEmpty() && !line.startsWith("#")) {
        id += 1

        // Make a new ExecutorService job for each sentence to parse.
        executorService.execute(new Runnable() {
          override def run() {
            val parses = parser.parseTokens(reader.readInput(line).getInputWords())
            val output = printer.printJointParses(parses, id)
            parsedSentences.getAndIncrement()
            printer synchronized {
              // It's a bit faster to buffer output than use
              // System.out.println() directly.
              sysout.write(output)
              sysout.newLine()
            }
          }
        })
      }
    }
    executorService.shutdown()
    executorService.awaitTermination(java.lang.Long.MAX_VALUE, TimeUnit.DAYS)
    sysout.close()

    val twoDP = new DecimalFormat("#.##")

    System.err.println("Sentences parsed: " + parsedSentences.get())
    System.err.println("Speed: "
                    + twoDP.format(1000.0 * parsedSentences.get() / timer.elapsed(TimeUnit.MILLISECONDS))
                    + " sentences per second")
  }

  def makePipelineParser(
    supertaggerBeam: Double,
    outputDependencies: Boolean
  ): PipelineSRLParser = {
    val posTagger = POSTagger.getStanfordTagger(new File(pipelineDir, "posTagger"))
    val labelClassifier = new File(pipelineDir, "labelClassifier")
    val classifier = if (labelClassifier.exists() && outputDependencies) Util.deserialize(labelClassifier) else CCGBankEvaluation.dummyLabelClassifier
    new PipelineSRLParser(makeParser(supertaggerBeam, 100000, false, None, defaultNBest, defaultMaxLength), classifier, posTagger)
  }

  def makeParser(
    supertaggerBeam: Double,
    maxChartSize: Int,
    joint: Boolean,
    supertaggerWeight: Option[Double],
    nbest: Int,
    maxLength: Int
  ): Parser = {
    Coindexation.parseMarkedUpFile(new File(modelDir, "markedup"))
    val cutoffsFile = new File(modelDir, "cutoffs")
    val cutoffs = if(cutoffsFile.exists()) Util.deserialize(cutoffsFile) else null

    val modelFactory = if (joint) {
      val keyToIndex = Util.deserialize[java.util.Map[FeatureKey, Integer]](new File(modelDir, "featureToIndex"))
      val weights = Util.deserialize[Array[Double]](new File(modelDir, "weights"))
      supertaggerWeight match {
        case Some(weight) => weights(0) = weight
        case None => { }
      }
      new SRLFactoredModelFactory(weights, Util.deserialize[FeatureSet](new File(modelDir, "features")).setSupertaggingFeature(new File(modelDir, "/pipeline"), supertaggerBeam), TaggerEmbeddings.loadCategories(new File(modelDir, "categories")), cutoffs, keyToIndex)
    } else {
      new SupertagFactoredModelFactory(Tagger.make(new File(modelDir), supertaggerBeam, 50, cutoffs), nbest > 1)
    }

    val parser = new ParserAStar(modelFactory, maxLength, nbest, rootCategories.map(Category.valueOf).asJava, new File(modelDir), maxChartSize)
    parser
  }
}

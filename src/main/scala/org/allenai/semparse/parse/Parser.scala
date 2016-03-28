package org.allenai.semparse.parse

import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.pipeline.Annotation
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import edu.stanford.nlp.parser.lexparser.LexicalizedParser
import edu.stanford.nlp.process.CoreLabelTokenFactory
import edu.stanford.nlp.process.PTBTokenizer
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation
import edu.stanford.nlp.trees.ModCollinsHeadFinder
import edu.stanford.nlp.trees.Tree
import edu.stanford.nlp.util.CoreMap

import java.io.StringReader
import java.util.Properties

import scala.collection.JavaConverters._

// A Parser takes sentence strings and returns parsed sentence strings.  It's just a thin
// abstraction layer over whatever parser you feel like using.
trait Parser {
  def parseSentence(sentence: String): ParsedSentence
}

// Some really simple representations, containing only what I need them to for the rest of this
// code.
trait ParsedSentence {
  def dependencies: Seq[Dependency]
  def tokens: Seq[Token]
  lazy val dependencyTree: DependencyTree = {
    val dependencyMap = dependencies.map(d => (d.headIndex, d)).groupBy(_._1).mapValues(_.map(_._2).sortBy(_.depIndex))
    val root = getNodeFromIndex(0, dependencyMap)
    root.children(0)._1
  }

  private def getNodeFromIndex(index: Int, dependencyMap: Map[Int, Seq[Dependency]]): DependencyTree = {
    val childDependencies = dependencyMap.getOrElse(index, Seq())
    val children = childDependencies.map(child => {
      val childNode = getNodeFromIndex(child.depIndex, dependencyMap)
      (childNode, child.label)
    })
    val token = if (index == 0) Token("ROOT", "ROOT", "ROOT", 0) else tokens(index-1)
    DependencyTree(token, children)
  }
}

class StanfordParser extends Parser {
  val props = new Properties()
  props.put("annotators","tokenize, ssplit, pos, lemma, parse")
  val pipeline = new StanfordCoreNLP(props)

  override def parseSentence(sentence: String): ParsedSentence = {
    val annotation = new Annotation(sentence)
    pipeline.annotate(annotation)
    val parsed = new StanfordParsedSentence(
      annotation.get(classOf[CoreAnnotations.SentencesAnnotation]).get(0))
    parsed
  }
}

class StanfordParsedSentence(sentence: CoreMap) extends ParsedSentence {
  override lazy val dependencies = {
    val deps = sentence.get(classOf[CollapsedCCProcessedDependenciesAnnotation]).typedDependencies
    deps.asScala.map(dependency => {
      Dependency(dependency.gov.label.value, dependency.gov.index,
        dependency.dep.label.value, dependency.dep.index, dependency.reln.toString)
    }).toSeq
  }

  override lazy val tokens = {
    val _tokens = sentence.get(classOf[CoreAnnotations.TokensAnnotation])
    _tokens.asScala.map(token => {
      val posTag = token.get(classOf[CoreAnnotations.PartOfSpeechAnnotation])
      val word = token.get(classOf[CoreAnnotations.TextAnnotation])
      val lemma = token.get(classOf[CoreAnnotations.LemmaAnnotation])
      Token(word, posTag, lemma, token.index)
    }).toSeq
  }
}

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
import scala.collection.mutable

// A Parser takes sentence strings and returns parsed sentence strings.  It's just a thin
// abstraction layer over whatever parser you feel like using.
trait Parser {
  def parseSentence(sentence: String): ParsedSentence
}

// Some really simple representations, containing only what I need them to for the rest of this
// code.
trait ParsedSentence {
  def dependencies: Seq[Dependency]
  def posTags: Seq[PartOfSpeech]
}

case class PartOfSpeech(word: String, posTag: String)
case class Dependency(head: String, headIndex: Int, dependent: String, depIndex: Int, label: String)

class StanfordParser(memoize: Boolean = true) extends Parser {
  val props = new Properties()
  props.put("annotators","tokenize, ssplit, pos, lemma, parse")
  val pipeline = new StanfordCoreNLP(props)

  val memoized = new mutable.HashMap[String, StanfordParsedSentence]

  override def parseSentence(sentence: String): ParsedSentence = {
    if (memoize && memoized.contains(sentence)) {
      memoized(sentence)
    } else {
      val annotation = new Annotation(sentence)
      pipeline.annotate(annotation)
      val parsed = new StanfordParsedSentence(
        annotation.get(classOf[CoreAnnotations.SentencesAnnotation]).get(0))
      if (memoize) {
        memoized.synchronized { memoized.update(sentence, parsed) }
      }
      parsed
    }
  }
}

class StanfordParsedSentence(sentence: CoreMap) extends ParsedSentence {
  override lazy val dependencies = {
    val deps = sentence.get(classOf[CollapsedCCProcessedDependenciesAnnotation]).typedDependencies
    deps.asScala.map(dependency => {
      Dependency(dependency.gov.backingLabel.value, dependency.gov.index,
        dependency.dep.backingLabel.value, dependency.dep.index, dependency.reln.toString)
    }).toSeq
  }

  override lazy val posTags = {
    val tokens = sentence.get(classOf[CoreAnnotations.TokensAnnotation])
    tokens.asScala.map(token => {
      val posTag = token.get(classOf[CoreAnnotations.PartOfSpeechAnnotation])
      val word = token.get(classOf[CoreAnnotations.TextAnnotation])
      PartOfSpeech(word, posTag)
    }).toSeq
  }
}


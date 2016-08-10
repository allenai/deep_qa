package org.allenai.dlfa.pipeline

import org.scalatest._

import org.json4s._
import org.json4s.JsonDSL._

import com.mattg.util.FileUtil

class LuceneBackgroundCorpusSearcherSpec extends FlatSpecLike with Matchers {
  val sentence = "This is a relatively long sentence that might contain useful information."
  val sentence2 = "This is a somewhat long sentence that might contain useful information."
  val sentence3 = "This is a slightly shorter sentence that might contain useful information."
  val different = "Not at all a similar sentence; this one is very different from the others."
  val different2 = "Another very different sentence containing all kinds of misinformation."
  val longSentence = "a b c " * 100

  val params: JValue =
    ("remove query near duplicates" -> true) ~
    ("sentences" ->
      ("sentence producer type" -> "sentence selector") ~
      ("create sentence indices" -> true) ~
      ("data name" -> "fake") ~
      ("data directory" -> "also fake"))
  val searcher = new LuceneBackgroundCorpusSearcher(params, new FileUtil)

  "consolidateHits" should "remove duplicates" in {
    val hits = Seq(sentence, sentence, sentence)
    searcher.consolidateHits("", hits, 10) should be(Seq(sentence))
  }

  it should "remove near duplicates" in {
    val hits = Seq(sentence, sentence2, sentence3, different)
    searcher.consolidateHits("", hits, 10) should be(Seq(sentence, different))
  }

  it should "remove near duplicates with the query" in {
    val positive = "Tiger sharks and killer whales eat adult sea turtles."
    val negative = "elephant sharks and killer whales eat adult sea turtles."
    val hits = Seq(positive, sentence)
    searcher.consolidateHits(negative, hits, 10) should be(Seq(sentence))
  }

  it should "keep only the top k" in {
    val hits = Seq(sentence, different, different2)
    searcher.consolidateHits("", hits, 2) should be(Seq(sentence, different))
  }

  it should "remove short results" in {
    val hits = Seq("short", "also short", "too short", "three is ok", sentence)
    searcher.consolidateHits("", hits, 10) should be(Seq("three is ok", sentence))
  }

  it should "remove newlines in results" in {
    val hits = Seq("has a\nnewline in it")
    searcher.consolidateHits("", hits, 10) should be(Seq("has a newline in it"))
  }

  it should "remove other newline variants" in {
    val hits = Seq("a\u0085b\rc\r\nd")
    searcher.consolidateHits("", hits, 10) should be(Seq("a b c d"))
  }

  it should "shorten whitespace" in {
    val hits = Seq("a      b              c                        d")
    searcher.consolidateHits("", hits, 10) should be(Seq("a b c d"))
  }

  it should "filter out the original query" in {
    val hits = Seq(sentence, different)
    searcher.consolidateHits(sentence, hits, 10) should be(Seq(different))
  }

  it should "remove long sentences" in {
    val hits = Seq(sentence, longSentence)
    searcher.consolidateHits("", hits, 10) should be(Seq(sentence))
  }
}

package org.allenai.semparse.pipeline.science_data

import org.scalatest._

import org.json4s._
import org.json4s.JsonDSL._

import com.mattg.util.FileUtil

class LuceneBackgroundCorpusSearcherSpec extends FlatSpecLike with Matchers {
  val sentence = "This is a relative long sentence that might contain useful information."
  val sentence2 = "This is a somewhat long sentence that might contain useful information."
  val sentence3 = "This is a slightly shorter sentence that might contain useful information."
  val different = "Not at all a similar sentence; this one is very different from the others."
  val different2 = "Another very different sentence containing all kinds of misinformation."

  "consolidateHits" should "remove duplicates" in {
    val hits = Seq(sentence, sentence, sentence)
    LuceneBackgroundCorpusSearcher.consolidateHits(hits, 10) should be(Seq(sentence))
  }

  "consolidateHits" should "remove near duplicates" in {
    val hits = Seq(sentence, sentence2, sentence3, different)
    LuceneBackgroundCorpusSearcher.consolidateHits(hits, 10) should be(Seq(sentence, different))
  }

  "consolidateHits" should "keep only the top k" in {
    val hits = Seq(sentence, different, different2)
    LuceneBackgroundCorpusSearcher.consolidateHits(hits, 2) should be(Seq(sentence, different))
  }

  "consolidateHits" should "remove short results" in {
    val hits = Seq("short", "also short", "too short", "three is ok", sentence)
    LuceneBackgroundCorpusSearcher.consolidateHits(hits, 10) should be(Seq("three is ok", sentence))
  }
}

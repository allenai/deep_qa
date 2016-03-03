package org.allenai.semparse.one_off

import org.scalatest._

import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods._

import edu.cmu.ml.rtw.users.matt.util.FakeFileUtil

class TestJsonAugmenterSpec extends FlatSpecLike with Matchers {
  implicit val formats = DefaultFormats

  "augmentLine" should "parse, augment, then output json" in {
    val augmenter = new TestJsonAugmenter {
      override def augmentJson(json: JValue): JValue = {
        (json \ "finished").extract[Boolean] should be(false)
        ("id" -> 1) ~ ("finished" -> true)
      }
    }
    val line = """{"finished": false}"""
    val expectedLine = """{"id":1,"finished":true}"""
    augmenter.augmentLine(line) should be(expectedLine)
  }

  "augmentJson" should "replace the queries json with new queries json" in {
    val augmenter = new TestJsonAugmenter {
      override def augmentQuery(queryJson: JValue): JValue = {
        val id = (queryJson \ "id").extract[Int]
        ("augmented id" -> id)
      }
    }
    val json: JValue = ("extra" -> true) ~ ("queries" -> List(("id" -> 1), ("id" -> 2), ("id" -> 3)))
    val expectedJson: JValue = ("extra" -> true) ~ ("queries" -> List(("augmented id" -> 1), ("augmented id" -> 2), ("augmented id" -> 3)))
    augmenter.augmentJson(json) should be(expectedJson)
  }

  "augmentQuery" should "add a midRelationsInQuery field to the query json" in {
    val augmenter = new TestJsonAugmenter {
      override def getWordRelsFromExpression(expression: String): Seq[(String, String, Boolean)] = {
        Seq(("word1", "mid1", false), ("word2", "mid2", true))
      }
    }
    val queryJson: JValue = ("queryExpression" -> "ignored for this test")
    val expectedJson: JValue = ("midRelationsInQuery" ->
      List(List(JString("word1"), JString("mid1"), JBool(false)), List(JString("word2"), JString("mid2"), JBool(true))))
    augmenter.augmentQuery(queryJson) should be(queryJson merge expectedJson)
  }

  "getWordRelsFromExpression" should "extract words and mids, and order" in {
    val augmenter = new TestJsonAugmenter
    augmenter.getWordRelsFromExpression("(lambda (var323865) (and ((word-cat \"original\") var323865) ((word-rel \"of\") \"/m/0424q8\" var323865)))") should be(
      Seq(("\"of\"", "\"/m/0424q8\"", true)))
    augmenter.getWordRelsFromExpression("(lambda (var286253) (and ((word-cat \"supermarket\") \"/m/02vg5b\") ((word-cat \"chain\") \"/m/02vg5b\") ((word-rel \"special:N/N\") var286253 \"/m/02vg5b\")))") should be(
      Seq(("\"special:N/N\"", "\"/m/02vg5b\"", false)))
    augmenter.getWordRelsFromExpression("(lambda (var338742) (and ((word-cat \"out\") var338742) ((word-rel \"of\") var338742 \"/m/01_d4\") ((word-rel \"owned_by\") \"/m/02mf93\" \"/m/03n23\")))") should be(
      Seq(("\"of\"", "\"/m/01_d4\"", false)))
    augmenter.getWordRelsFromExpression("(lambda (x) (and ((word-rel \"owned_by\") \"/m/02mf93\" x)))") should be(
      Seq(("\"owned_by\"", "\"/m/02mf93\"", true)))
    augmenter.getWordRelsFromExpression("(lambda (x) (and ((word-rel \"owned_by\") \"/m/02mf93\" x) ((word-rel \"test\") \"test-mid\" x)))") should be(
      Seq(("\"owned_by\"", "\"/m/02mf93\"", true), ("\"test\"", "\"test-mid\"", true)))
  }
}

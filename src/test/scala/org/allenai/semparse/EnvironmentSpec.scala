package org.allenai.semparse

import org.scalatest._

import edu.cmu.ml.rtw.users.matt.util.FileUtil

class EnvironmentSpec extends FlatSpecLike with Matchers {

  "evalulateSExpression" should "correctly evaluate a simple expression" in {
    val env = new Environment(Seq(), Seq())
    env.evalulateSExpression("3").getValue() should be(3)
    env.evalulateSExpression("(+ 3 5)").getValue() should be(8)
  }

  it should "pass in extra arguments correctly" in {
    val env = new Environment(Seq(), Seq("extra argument"))
    env.evalulateSExpression("(car ARGV)").getValue() should be("extra argument")
  }

  it should "load passed in files and update the environment from them" in {
    val fileUtil = new FileUtil
    val lispFilename = "test_file.lisp"
    val lispLines = Seq("(define squareTest (x) (* x  x))")
    val secondLispFilename = "test_file2.lisp"
    val secondLispLines = Seq("(define squareTest2 (squareTest 4))")
    fileUtil.writeLinesToFile(lispFilename, lispLines)
    fileUtil.writeLinesToFile(secondLispFilename, secondLispLines)

    val env = new Environment(Seq(lispFilename, secondLispFilename), Seq())
    env.evalulateSExpression("(squareTest 3)").getValue() should be(9)
    env.evalulateSExpression("squareTest2").getValue() should be(16)

    fileUtil.deleteFile(lispFilename)
    fileUtil.deleteFile(secondLispFilename)
  }

  it should "take previously bound names into account" in {
    val env = new Environment(Seq(), Seq())
    env.bindName("test", Integer.valueOf(3))
    env.evalulateSExpression("test").getValue() should be(3)
  }
}

package org.allenai.semparse

import scala.collection.JavaConverters._

import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.lisp.AmbEval
import com.jayantkrish.jklol.lisp.ConsValue
import com.jayantkrish.jklol.lisp.LispUtil
import com.jayantkrish.jklol.lisp.ParametricBfgBuilder
import com.jayantkrish.jklol.lisp.SExpression

// This is really similar to AmbLisp in jklol.lisp.cli.  The main difference is that AmbLisp runs a
// lisp program, perhaps leaving you in interactive mode where you can input stuff through stdin.
// Here, we can similarly run lisp programs, but we keep the environment around for you to
// programmatically evaluate SExpressions.  Much more handy, I think, for running evaluations, than
// trying to go through stdin and stdout with AmbLisp.
class Environment(lispFiles: Seq[String], extraArgs: Seq[String], verbose: Boolean = false) {
  val symbolTable = AmbEval.getInitialSymbolTable()
  val eval = new AmbEval(symbolTable)
  val parser = ExpressionParser.sExpression(symbolTable)
  if (verbose) println(s"Reading program files: $lispFiles")
  val programExpression = LispUtil.readProgram(lispFiles.asJava, symbolTable)
  val fgBuilder = new ParametricBfgBuilder(true)
  val environment = {
    val env = AmbEval.getDefaultEnvironment(symbolTable)
    env.bindName(AmbEval.CLI_ARGV_VAR_NAME, ConsValue.listToConsList(extraArgs.asJava), symbolTable)
    env
  }

  if (verbose) println("Loading initial environment")
  eval.eval(programExpression, environment, fgBuilder)

  def evalulateSExpression(expressionText: String) = {
    val expression = parser.parse(expressionText)
    eval.eval(expression, environment, new ParametricBfgBuilder(true))
  }

  def evalulateSExpression(expression: SExpression) = {
    eval.eval(expression, environment, new ParametricBfgBuilder(true))
  }

  def bindName(name: String, value: Object) {
    environment.bindName(name, value, symbolTable)
  }
}

package org.allenai.semparse

import scala.collection.JavaConverters._

import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.lisp.AmbEval
import com.jayantkrish.jklol.lisp.ConsValue
import com.jayantkrish.jklol.lisp.{Environment => JEnv}
import com.jayantkrish.jklol.lisp.LispUtil
import com.jayantkrish.jklol.lisp.ParametricBfgBuilder
import com.jayantkrish.jklol.lisp.SExpression
import com.jayantkrish.jklol.util.IndexedList

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
    bindBuiltinFunctions(env, symbolTable)
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

  // This defines some functions that we can use in lisp code without defining them there.
  def bindBuiltinFunctions(env: JEnv, symbolTable: IndexedList[String]) {
    env.bindName("create-sfe-feature-computer", new CreateSfeFeatureComputer(), symbolTable);
    env.bindName("display-parameters", new DisplayParameters(), symbolTable);
    env.bindName("find-related-entities-in-graph", new FindRelatedEntities(), symbolTable);
    env.bindName("get-cat-word-feature-list", new GetCatWordFeatureList(), symbolTable);
    env.bindName("get-entity-feature-difference", new GetEntityFeatureDifference(), symbolTable);
    env.bindName("get-entity-features", new GetEntityFeatures(), symbolTable);
    env.bindName("get-entity-tuple-feature-difference", new GetEntityPairFeatureDifference(), symbolTable);
    env.bindName("get-entity-tuple-features", new GetEntityPairFeatures(), symbolTable);
    env.bindName("get-rel-word-feature-list", new GetRelWordFeatureList(), symbolTable);
  }
}

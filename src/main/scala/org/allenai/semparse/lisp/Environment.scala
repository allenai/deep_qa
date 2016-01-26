package org.allenai.semparse.lisp

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

object Environment {
  val BASE_ENVIRONMENT_FILE = "src/lisp/environment.lisp"
  val USCHEMA_ENVIRONMENT_FILE = "src/lisp/uschema_environment.lisp"

  val DISTRIBUTIONAL_MODEL_FILE = "src/lisp/model.lisp"
  val GRAPH_MODEL_FILE = "src/lisp/model_with_graphs.lisp"
  val BASELINE_MODEL_FILE = "src/lisp/baseline_model.lisp"

  val PREDICATE_RANKING_FILE = "src/lisp/predicate_ranking.lisp"
  val QUERY_RANKING_FILE = "src/lisp/query_ranking.lisp"
  val TRAIN_FILE = "src/lisp/train_model.lisp"

  val EVAL_BASELINE_FILE = "src/lisp/eval_baseline.lisp"
  val EVAL_USCHEMA_FILE = "src/lisp/eval_uschema.lisp"
  val EVAL_ENSEMBLE_FILE = "src/lisp/eval_ensemble.lisp"

  val SMALL_WORD_FILE = "data/small/words.lisp"
  val SMALL_ENTITY_FILE = "data/small/entities.lisp"
  val SMALL_JOINT_ENTITY_FILE = "data/small/joint_entities.lisp"
  val SMALL_PREDICATE_RANKING_LF_FILE = "data/small/predicate_ranking_lf.lisp"
  val SMALL_QUERY_RANKING_LF_FILE = "data/small/query_ranking_lf.lisp"
  val SMALL_SFE_SPEC_FILE = "data/small/sfe_spec.json"

  val LARGE_WORD_FILE = "data/large/words.lisp"
  val LARGE_ENTITY_FILE = "data/large/entities.lisp"
  val LARGE_JOINT_ENTITY_FILE = "data/large/joint_entities.lisp"
  val LARGE_PREDICATE_RANKING_LF_FILE = "data/large/predicate_ranking_lf.lisp"
  val LARGE_QUERY_RANKING_LF_FILE = "data/large/query_ranking_lf.lisp"
  val LARGE_SFE_SPEC_FILE = "data/large/sfe_spec.json"

  val COMMON_SMALL_DATA = Seq(SMALL_ENTITY_FILE, SMALL_WORD_FILE)
  val PREDICATE_RANKING_SMALL_DATA = COMMON_SMALL_DATA ++ Seq(SMALL_PREDICATE_RANKING_LF_FILE)
  val QUERY_RANKING_SMALL_DATA = COMMON_SMALL_DATA ++ Seq(
    SMALL_JOINT_ENTITY_FILE, SMALL_QUERY_RANKING_LF_FILE)

  val COMMON_LARGE_DATA = Seq(LARGE_ENTITY_FILE, LARGE_WORD_FILE)
  val PREDICATE_RANKING_LARGE_DATA = COMMON_LARGE_DATA ++ Seq(LARGE_PREDICATE_RANKING_LF_FILE)
  val QUERY_RANKING_LARGE_DATA = COMMON_LARGE_DATA ++ Seq(
    LARGE_JOINT_ENTITY_FILE, LARGE_QUERY_RANKING_LF_FILE)

  val ENV_FILES = Seq(BASE_ENVIRONMENT_FILE, USCHEMA_ENVIRONMENT_FILE)

  val TRAIN_QUERY_RANKING_DISTRIBUTIONAL_MODEL = ENV_FILES ++ Seq(
    QUERY_RANKING_FILE, DISTRIBUTIONAL_MODEL_FILE, TRAIN_FILE)
  val TRAIN_PREDICATE_RANKING_DISTRIBUTIONAL_MODEL = ENV_FILES ++ Seq(
    PREDICATE_RANKING_FILE, DISTRIBUTIONAL_MODEL_FILE, TRAIN_FILE)

  val TRAIN_QUERY_RANKING_GRAPH_MODEL = ENV_FILES ++ Seq(
    QUERY_RANKING_FILE, GRAPH_MODEL_FILE, TRAIN_FILE)
  val TRAIN_PREDICATE_RANKING_GRAPH_MODEL = ENV_FILES ++ Seq(
    PREDICATE_RANKING_FILE, GRAPH_MODEL_FILE, TRAIN_FILE)
}

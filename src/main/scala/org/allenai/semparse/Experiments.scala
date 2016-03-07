package org.allenai.semparse

// This defines the experiments that we want to run, which is used by the main methods in Trainer
// and Evaluator.  It also keeps pointers to all of the lisp input files and data files that we
// use, and groups some of them together in useful ways.
object Experiments {

  // Currently, there are four options in the experiments: which dataset do you use, ensembled with
  // baseline or not (for evaluation only), what ranking method, and what model type.  If we add
  // very many more possibilities, this will need some re-thinking to be more sane.
  val experimentConfigs: Seq[(String, String, String, Boolean)] = {
    val datasets = Seq("large")//, "small")
    val modelTypes = Seq("formal")//, "distributional", "combined")

    // I don't actually change these in the experiments for the ACL 2016 paper, but these were
    // options used for the TACL 2015 paper.
    val rankings = Seq("query")//, "predicate")
    val ensembledEvaluation = Seq(false)//, true)
    val baselineConfigs = Seq(
      //("small", "baseline", "ignored", false)
      //("large", "baseline", "ignored", false)
    )

    // Generate all possible combinations
    val configs = for (data <- datasets;
         modelType <- modelTypes;
         ranking <- rankings;
         ensembled <- ensembledEvaluation) yield (data, modelType, ranking, ensembled)

    (configs ++ baselineConfigs)
  }

  val BASE_ENVIRONMENT_FILE = "src/main/lisp/environment.lisp"
  val USCHEMA_ENVIRONMENT_FILE = "src/main/lisp/uschema_environment.lisp"

  val DISTRIBUTIONAL_MODEL_FILE = "src/main/lisp/model_distributional.lisp"
  val COMBINED_MODEL_FILE = "src/main/lisp/model_combined.lisp"
  val FORMAL_MODEL_FILE = "src/main/lisp/model_formal.lisp"
  val BASELINE_MODEL_FILE = "src/main/lisp/model_baseline.lisp"

  val PREDICATE_RANKING_FILE = "src/main/lisp/predicate_ranking.lisp"
  val QUERY_RANKING_FILE = "src/main/lisp/query_ranking.lisp"
  val TRAIN_FILE = "src/main/lisp/train_model.lisp"

  val EVAL_BASELINE_FILE = "src/main/lisp/eval_baseline.lisp"
  val EVAL_USCHEMA_FILE = "src/main/lisp/eval_uschema.lisp"
  val EVAL_ENSEMBLE_FILE = "src/main/lisp/eval_ensemble.lisp"

  val SMALL_BASE_DATA_FILE = "data/tacl2015-training-sample.txt"
  val SMALL_WORD_FILE = "data/small/words.lisp"
  val SMALL_ENTITY_FILE = "data/small/entities.lisp"
  val SMALL_JOINT_ENTITY_FILE = "data/small/joint_entities.lisp"
  val SMALL_PREDICATE_RANKING_LF_FILE = "data/small/predicate_ranking_lf.lisp"
  val SMALL_QUERY_RANKING_LF_FILE = "data/small/query_ranking_lf.lisp"
  val SMALL_SFE_SPEC_FILE = "src/main/resources/sfe_spec_small.json"

  val LARGE_BASE_DATA_FILE = "data/acl2016-training.txt"
  val LARGE_WORD_FILE = "data/large/words.lisp"
  val LARGE_ENTITY_FILE = "data/large/entities.lisp"
  val LARGE_JOINT_ENTITY_FILE = "data/large/joint_entities.lisp"
  val LARGE_PREDICATE_RANKING_LF_FILE = "data/large/predicate_ranking_lf.lisp"
  val LARGE_QUERY_RANKING_LF_FILE = "data/large/query_ranking_lf.lisp"
  val LARGE_SFE_SPEC_FILE = "src/main/resources/sfe_spec_large.json"

  val TEST_DATA_FILE = "src/main/resources/acl2016_final_test_set_annotated.json"

  val COMMON_SMALL_DATA = Seq(SMALL_ENTITY_FILE, SMALL_WORD_FILE)
  val PREDICATE_RANKING_SMALL_DATA = COMMON_SMALL_DATA ++ Seq(SMALL_PREDICATE_RANKING_LF_FILE)
  val QUERY_RANKING_SMALL_DATA = COMMON_SMALL_DATA ++ Seq(
    SMALL_JOINT_ENTITY_FILE, SMALL_QUERY_RANKING_LF_FILE)

  val COMMON_LARGE_DATA = Seq(LARGE_ENTITY_FILE, LARGE_WORD_FILE)
  val PREDICATE_RANKING_LARGE_DATA = COMMON_LARGE_DATA ++ Seq(LARGE_PREDICATE_RANKING_LF_FILE)
  val QUERY_RANKING_LARGE_DATA = COMMON_LARGE_DATA ++ Seq(
    LARGE_JOINT_ENTITY_FILE, LARGE_QUERY_RANKING_LF_FILE)

  val ENV_FILES = Seq(BASE_ENVIRONMENT_FILE, USCHEMA_ENVIRONMENT_FILE)
}

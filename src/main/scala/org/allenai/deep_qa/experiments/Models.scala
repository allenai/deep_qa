package org.allenai.deep_qa.experiments

import org.json4s._
import org.json4s.JsonDSL._

/**
 * This is just a collection of JValue specifications for various kinds of models.
 */
object Models {
  val multipleTrueFalseMemoryNetwork: JValue =
    ("model_class" -> "MultipleTrueFalseMemoryNetwork")

  val softmaxMemoryNetwork: JValue =
    ("model_class" -> "SoftmaxMemoryNetwork")

  val questionAnswerMemoryNetwork: JValue =
    ("model_class" -> "QuestionAnswerMemoryNetwork")

  val trueFalseMemoryNetwork: JValue =
    ("model_class" -> "MemoryNetwork")

  val basicMemoryNetworkComponents: JValue =
    ("knowledge_selector" -> ("type" -> "dot_product")) ~
    ("memory_updater" -> ("type" -> "sum")) ~
    ("entailment_input_combiner" -> ("type" -> "memory_only")) ~
    ("num_memory_layers" -> 1)

  val betterEntailmentComponents: JValue =
    ("knowledge_selector" -> ("type" -> "dot_product")) ~
    ("memory_updater" -> ("type" -> "dense_concat")) ~
    ("entailment_input_combiner" -> ("type" -> "heuristic_matching")) ~
    ("num_memory_layers" -> 1)

  val decomposableAttentionSolver: JValue =
    ("model_class" -> "DecomposableAttention")

  val multipleTrueFalseDecomposableAttentionSolver: JValue =
    ("model_class" -> "MultipleTrueFalseDecomposableAttention")

  def endToEndMemoryNetwork(encoder: String, numMemoryLayers: Int): JValue =
    softmaxMemoryNetwork merge
    basicMemoryNetworkComponents merge
    ("knowledge_encoder" -> ("type" -> "temporal")) ~
    ("embedding_size" -> 20) ~
    ("encoder" -> ("type" -> encoder)) ~
    ("num_memory_layers" -> numMemoryLayers)

  val attentionSumReader: JValue =
    ("model_class" -> "AttentionSumReader") ~
    ("patience" -> 1) ~
    ("preferred_backend" -> "theano") ~
    ("encoder" ->
      ("default" ->
        ("type" -> "bi_gru")~
        ("output_dim" -> 384)
      )
    ) ~
    ("seq2seq_encoder" ->
      ("default" ->
        ("type" -> "bi_gru") ~
        ("encoder_params" ->
          ("output_dim" -> 384)
        ) ~
        ("wrapper_params" -> JObject())
      )
    ) ~
    ("optimizer" ->
      ("type" -> "adam") ~
      ("clipnorm" -> 10.0) ~
      ("lr" -> 0.0005)
    ) ~
    ("embedding_dropout" -> 0.0) ~
    ("patience" -> 0) ~
    ("embedding_size" -> 256) ~
    ("num_epochs" -> 5)

  val gatedAttentionReader: JValue =
    ("model_class" -> "GatedAttentionReader") ~
    ("embedding_size" -> 100) ~
    ("pretrained_embeddings_file" -> "/efs/data/dlfa/glove/glove.6B.100d.txt.gz") ~
    ("fine_tune_embeddings" -> false) ~
    ("project_embeddings" -> false) ~
    ("num_gated_attention_layers" -> 3) ~
    ("patience" -> 0) ~
    ("preferred_backend" -> "theano") ~
    ("encoder" ->
      ("question_final" ->
        ("type" -> "bi_gru")~
        ("output_dim" -> 128)
      )
    ) ~
    ("seq2seq_encoder" ->
      ("question_0" ->
        ("type" -> "bi_gru") ~
        ("encoder_params" ->
          ("output_dim" -> 128)
        ) ~
        ("wrapper_params" -> JObject())
      ) ~
      ("document_0" ->
        ("type" -> "bi_gru") ~
        ("encoder_params" ->
          ("output_dim" -> 128)
        ) ~
        ("wrapper_params" -> JObject())
      ) ~
      ("question_1" ->
        ("type" -> "bi_gru") ~
        ("encoder_params" ->
          ("output_dim" -> 128)
        ) ~
        ("wrapper_params" -> JObject())
      ) ~
      ("document_1" ->
        ("type" -> "bi_gru") ~
        ("encoder_params" ->
          ("output_dim" -> 128)
        ) ~
        ("wrapper_params" -> JObject())
      ) ~
      ("document_final" ->
        ("type" -> "bi_gru") ~
        ("encoder_params" ->
          ("output_dim" -> 128)
        ) ~
        ("wrapper_params" -> JObject())
      )
    ) ~
    ("optimizer" ->
      ("type" -> "adam") ~
      ("clipnorm" -> 10.0) ~
      ("lr" -> 0.0005)
    ) ~
    ("embedding_dropout" -> 0.0) ~
    ("patience" -> 0) ~
    ("num_epochs" -> 5)
}

object Debug {
  val multipleTrueFalseDefault: JValue =
    ("debug" ->
      ("layer_names" -> List(
        "timedist_knowledge_selector_0",
        "entailment_scorer",
        "answer_option_softmax")) ~
      ("data" -> "training"))
}

object Encoders {
  def encoder(encoderType: String, extraParams: JValue=JNothing): JValue = {
    val encoderParams = (("type" -> encoderType): JValue) merge extraParams
    ("encoder" -> encoderParams)
  }

  val bagOfWords = encoder("bow")
  val lstm = encoder("lstm")
  val gru = encoder("gru")
  val fourGramCnn_50filters = encoder(
    "cnn",
    ("ngram_filter_sizes" -> List(2, 3, 4)) ~ ("num_filters" -> 50)
  )

  def pretrainedGloveEmbeddings(fineTuning: Boolean=true, projection: Boolean=false): JValue =
    ("pretrained_embeddings_file" -> "/efs/data/dlfa/glove.840B.300d.txt.gz") ~
    ("fine_tune_embeddings" -> fineTuning) ~
    ("project_embeddings" -> projection)
}

object Training {
  // This is just for making sure things actually run.
  val fastTest: JValue =
    ("max_training_instances" -> 10) ~
    ("num_epochs" -> 1)

  val default: JValue =
    ("num_epochs" -> 30) ~
    ("patience" -> 5)

  val long: JValue =
    ("num_epochs" -> 100) ~
    ("patience" -> 10)

  val entailmentPretraining: JValue =
    ("pretrainers" -> List(
      ("type" -> "SnliEntailmentPretrainer") ~
      ("num_epochs" -> 40) ~
      ("patience" -> 1) ~
      ("train_files" -> List("/efs/data/dlfa/table_mcq_test/snli_training_data.tsv"))
    ))

  val attentionPretraining: JValue =
    ("pretrainers" -> List(
      ("type" -> "AttentionPretrainer") ~
      ("num_epochs" -> 40) ~
      ("patience" -> 1) ~
      ("train_files" -> List(
        "/efs/data/dlfa/table_mcq_test/training_data.tsv",
        "/efs/data/dlfa/table_mcq_test/test_data_labeled_background.tsv"))
    ))

  val encoderPretraining: JValue =
    ("pretrainers" -> List(
      ("type" -> "EncoderPretrainer") ~
      ("num_epochs" -> 40) ~
      ("patience" -> 1) ~
      ("train_files" -> List("/efs/data/dlfa/busc/busc_encoder_pretrain_data_shuf.tsv"))
    ))
}

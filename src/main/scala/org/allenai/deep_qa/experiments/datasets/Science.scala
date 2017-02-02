package org.allenai.deep_qa.experiments.datasets

import org.json4s._
import org.json4s.JsonDSL._

/**
 * This is a collection of DatasetStep specifications, for easy re-use in various experiments.
 */
object ScienceDatasets {
  def makeBackgroundDataset(
    sentences: JValue,
    sentenceFormat: String,
    corpus: JValue
  ): JValue = {
    ("data files" -> List(
      sentences,
      ("sentence producer type" -> "background searcher") ~
      ("searcher" -> corpus) ~
      ("sentences" -> sentences) ~
      ("sentence format" -> sentenceFormat)))
  }

  def makeCombinedDataset(datasets: Seq[JValue], outputDirectory: String): JValue = {
    ("dataset type" -> "combined") ~
    ("datasets" -> datasets) ~
    ("output directory" -> outputDirectory)
  }

  val omnibusMtfGradeFourTrainQuestionsWithBuscBackground: JValue = makeBackgroundDataset(
    ScienceFiles.omnibusGradeFourTrainSentences_multipleTrueFalse_appendAnswer,
    "plain sentence",
    ScienceCorpora.buscElasticSearchIndex(10)
  )

  val omnibusMtfGradeFourDevQuestionsWithBuscBackground: JValue = makeBackgroundDataset(
    ScienceFiles.omnibusGradeFourDevSentences_multipleTrueFalse_appendAnswer,
    "plain sentence",
    ScienceCorpora.buscElasticSearchIndex(10)
  )

  val omnibusMtfGradeEightTrainQuestionsWithBuscBackground: JValue = makeBackgroundDataset(
    ScienceFiles.omnibusGradeEightTrainSentences_multipleTrueFalse_appendAnswer,
    "plain sentence",
    ScienceCorpora.buscElasticSearchIndex(10)
  )

  val omnibusMtfGradeEightDevQuestionsWithBuscBackground: JValue = makeBackgroundDataset(
    ScienceFiles.omnibusGradeEightDevSentences_multipleTrueFalse_appendAnswer,
    "plain sentence",
    ScienceCorpora.buscElasticSearchIndex(10)
  )

  val omnibusMtfGradeFourAndEightTrainQuestionsWithBuscBackground: JValue = makeCombinedDataset(Seq(
    omnibusMtfGradeFourTrainQuestionsWithBuscBackground,
    omnibusMtfGradeEightTrainQuestionsWithBuscBackground
  ), "/efs/data/dlfa/processed/omnibus_4_8_train/")

  val omnibusMtfGradeFourAndEightDevQuestionsWithBuscBackground: JValue = makeCombinedDataset(Seq(
    omnibusMtfGradeFourDevQuestionsWithBuscBackground,
    omnibusMtfGradeEightDevQuestionsWithBuscBackground
  ), "/efs/data/dlfa/processed/omnibus_4_8_dev/")

  val diagramMtfQuestionsWithBuscBackground: JValue = makeBackgroundDataset(
    ScienceFiles.ai2diagramSentences_multipleTrueFalse_appendAnswer,
    "plain sentence",
    ScienceCorpora.buscElasticSearchIndex(10)
  )

  val openQaMtfQuestionsWithBuscBackground: JValue = makeBackgroundDataset(
    ScienceFiles.openQaAnimalAndGeneralScience_multipleTrueFalse_appendAnswer,
    "plain sentence",
    ScienceCorpora.buscElasticSearchIndex(10))

  val omnibusQaGradeFourTrainQuestionsWithBuscBackground: JValue = makeBackgroundDataset(
    ScienceFiles.omnibusGradeFourTrainSentences_questionAndAnswer,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )

  val omnibusQaGradeFourDevQuestionsWithBuscBackground: JValue = makeBackgroundDataset(
    ScienceFiles.omnibusGradeFourDevSentences_questionAndAnswer,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )

  val omnibusQaGradeEightTrainQuestionsWithBuscBackground: JValue = makeBackgroundDataset(
    ScienceFiles.omnibusGradeEightTrainSentences_questionAndAnswer,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )

  val omnibusQaGradeEightDevQuestionsWithBuscBackground: JValue = makeBackgroundDataset(
    ScienceFiles.omnibusGradeEightDevSentences_questionAndAnswer,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )

  val diagramQaQuestionsWithBuscBackground: JValue = makeBackgroundDataset(
    ScienceFiles.ai2diagramSentences_questionAndAnswer,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )

  val tableMcqTrain: JValue =
    ("sentence producer type" -> "manually provided") ~
    ("create sentence indices" -> true) ~
    ("filename" -> "/efs/data/dlfa/table_mcq_test/training_data.tsv")

  val tableMcqDev: JValue =
    ("sentence producer type" -> "manually provided") ~
    ("create sentence indices" -> true) ~
    ("filename" -> "/efs/data/dlfa/table_mcq_test/validation_data.tsv")

  val tableMcqCorrectBackground: JValue =
    ("sentence producer type" -> "manually provided") ~
    ("create sentence indices" -> true) ~
    ("filename" -> "/efs/data/dlfa/table_mcq_test/test_data_background_correct_only.tsv")

  val tableMcqTenBackground: JValue =
    ("sentence producer type" -> "manually provided") ~
    ("create sentence indices" -> true) ~
    ("filename" -> "/efs/data/dlfa/table_mcq_test/test_data_background.tsv")

  val tableMcqTrainWithCorrectBackground: JValue = ("data files" -> List(tableMcqTrain, tableMcqCorrectBackground))
  val tableMcqDevWithCorrectBackground: JValue = ("data files" -> List(tableMcqDev, tableMcqCorrectBackground))
  val tableMcqTrainWithTenBackground: JValue = ("data files" -> List(tableMcqTrain, tableMcqTenBackground))
  val tableMcqDevWithTenBackground: JValue = ("data files" -> List(tableMcqDev, tableMcqTenBackground))
}

/**
 * These are individual files that make up Datasets.  Because these sometimes depend on each other,
 * we can't really specify them all declaratively in Datasets, so we specify them here first, then
 * use them above in Datasets.
 */
object ScienceFiles {
  def makeMultipleTrueFalseQuestionAnswerFile(questionFile: String, outputFile: String): JValue = {
    ("sentence producer type" -> "question interpreter") ~
    ("create sentence indices" -> true) ~
    ("question file" -> questionFile) ~
    ("output file" -> outputFile) ~
    ("interpreter" -> ("type" -> "append answer"))
  }

  def makeQuestionAnswerFile(questionFile: String, outputFile: String): JValue = {
    ("sentence producer type" -> "question interpreter") ~
    ("create sentence indices" -> true) ~
    ("question file" -> questionFile) ~
    ("output file" -> outputFile) ~
    ("interpreter" -> ("type" -> "question and answer"))
  }

  val omnibusGradeFourTrainSentences_multipleTrueFalse_appendAnswer: JValue =
    makeMultipleTrueFalseQuestionAnswerFile(
      "/efs/data/dlfa/questions/omnibus_4_train.tsv",
      "/efs/data/dlfa/processed/omnibus_4_train/multiple_tf/append_answer/sentences.tsv"
    )

  val omnibusGradeFourDevSentences_multipleTrueFalse_appendAnswer: JValue =
    makeMultipleTrueFalseQuestionAnswerFile(
      "/efs/data/dlfa/questions/omnibus_4_dev.tsv",
      "/efs/data/dlfa/processed/omnibus_4_dev/multiple_tf/append_answer/sentences.tsv"
    )

  val omnibusGradeEightTrainSentences_multipleTrueFalse_appendAnswer: JValue =
    makeMultipleTrueFalseQuestionAnswerFile(
      "/efs/data/dlfa/questions/omnibus_8_train.tsv",
      "/efs/data/dlfa/processed/omnibus_8_train/multiple_tf/append_answer/sentences.tsv"
    )

  val omnibusGradeEightDevSentences_multipleTrueFalse_appendAnswer: JValue =
    makeMultipleTrueFalseQuestionAnswerFile(
      "/efs/data/dlfa/questions/omnibus_8_dev.tsv",
      "/efs/data/dlfa/processed/omnibus_8_dev/multiple_tf/append_answer/sentences.tsv"
    )

  val ai2diagramSentences_multipleTrueFalse_appendAnswer: JValue =
    makeMultipleTrueFalseQuestionAnswerFile(
      "/efs/data/dlfa/questions/diagram_questions.tsv",
      "/efs/data/dlfa/processed/diagram_questions/multiple_tf/append_answer/sentences.tsv"
    )

  val omnibusGradeFourTrainSentences_questionAndAnswer: JValue =
    makeQuestionAnswerFile(
      "/efs/data/dlfa/questions/omnibus_4_train.tsv",
      "/efs/data/dlfa/processed/omnibus_4_train/question_and_answer/questions.tsv"
    )

  val omnibusGradeFourDevSentences_questionAndAnswer: JValue =
    makeQuestionAnswerFile(
      "/efs/data/dlfa/questions/omnibus_4_dev.tsv",
      "/efs/data/dlfa/processed/omnibus_4_dev/question_and_answer/questions.tsv"
    )

  val omnibusGradeEightTrainSentences_questionAndAnswer: JValue =
    makeQuestionAnswerFile(
      "/efs/data/dlfa/questions/omnibus_8_train.tsv",
      "/efs/data/dlfa/processed/omnibus_8_train/question_and_answer/questions.tsv"
    )

  val omnibusGradeEightDevSentences_questionAndAnswer: JValue =
    makeQuestionAnswerFile(
      "/efs/data/dlfa/questions/omnibus_8_dev.tsv",
      "/efs/data/dlfa/processed/omnibus_8_dev/question_and_answer/questions.tsv"
    )

  val ai2diagramSentences_questionAndAnswer: JValue =
    makeQuestionAnswerFile(
      "/efs/data/dlfa/questions/diagram_questions.tsv",
      "/efs/data/dlfa/processed/diagram_questions/question_and_answer/questions.tsv"
    )

  val openQaAnimalAndGeneralScience_multipleTrueFalse_appendAnswer: JValue =
    ("sentence producer type" -> "dataset reader") ~
      ("reader" -> "open qa") ~
      ("create sentence indices" -> true) ~
      ("input file" -> "/efs/data/dlfa/open_qa/science_related.json") ~
      ("output files" ->
        Seq("/efs/data/dlfa/processed/open_qa/questions.tsv"))
}

/**
 * These are corpora that can be used for searching, or for training language models, or the like.
 */
object ScienceCorpora {
  def buscElasticSearchIndex(numResults: Int): JValue =
    ("num passages per query" -> numResults) ~
    ("elastic search index url" -> "aristo-es1.dev.ai2") ~
    ("elastic search index port" -> 9300) ~
    ("elastic search cluster name" -> "aristo-es") ~
    ("elastic search index name" -> Seq("busc"))

  def aristoDefaultElasticSearchIndex(numResults: Int): JValue =
    ("num passages per query" -> numResults) ~
    ("elastic search index url" -> "aristo-es1.dev.ai2") ~
    ("elastic search index port" -> 9300) ~
    ("elastic search cluster name" -> "aristo-es") ~
    ("elastic search index name" -> Seq(
      "barrons",
      "websentences",
      "ck12biov44",
      "waterloo",
      "wikipedia",
      "simplewikipedia")
  )
}

/**
 * This is a private object because it contains parameters for _creating_ the files.  It's also
 * horribly out of date and not in use anymore.  Once we have a nice process around this that
 * actually works, we should clean this up somehow.
 */
object CreatedScienceDatasets {
  //////////////////////////////////////////////////////////
  // Step 1: Take a corpus and select sentences to use
  //////////////////////////////////////////////////////////

  def johannesBackgroundDataset(version: String): JValue = {
    ScienceDatasets.makeBackgroundDataset(
      ("sentence producer type" -> "manually provided") ~
        ("create sentence indices" -> true) ~
        ("filename" -> s"/efs/data/dlfa/generated_questions/v${version}/sentences.tsv"),
      "plain sentence",
      ScienceCorpora.buscElasticSearchIndex(10) merge (("remove query near duplicates" -> true): JValue)
    )
  }

  val corpus = "s3n://private.store.dev.allenai.org/org.allenai.corpora.busc/extractedDocuments/science_templates"
  val sentenceSelectorParams: JValue =
    ("sentence producer type" -> "sentence selector") ~
    ("create sentence indices" -> true) ~
    ("sentence selector" -> ("max word count per sentence" -> 20) ~
                            ("min word count per sentence" -> 6)) ~
    ("data name" -> "busc") ~
    ("data directory" -> corpus) ~
    ("max sentences" -> 250000)

  //////////////////////////////////////////////////////////////////
  // Step 2: Corrupt the positive sentences to get negative data
  //////////////////////////////////////////////////////////////////

  // Step 2a: train a language model on the positive data.
  val languageModelParams: JValue =
    ("sentences" -> sentenceSelectorParams) ~
    ("tokenize input" -> false) ~
    ("use lstm" -> true) ~
    ("word dimensionality" -> 50) ~
    ("max training epochs" -> 20)

  // Step 2b: generate candidate corruptions using the KB
  val kbSentenceCorruptorParams: JValue =
    ("positive data" -> sentenceSelectorParams) ~
    ("kb tensor file" -> "/home/mattg/data/aristo_kb/animals-tensor-july10-yesonly-wo-thing.csv")

  // Step 2c: use the language model to select among the candidates
  val corruptedSentenceSelectorParams: JValue =
    ("sentence producer type" -> "kb sentence corruptor") ~
    ("create sentence indices" -> true) ~
    ("candidates per set" -> 5) ~
    ("max sentences" -> 250000) ~
    ("corruptor" -> kbSentenceCorruptorParams) ~
    ("language model" -> languageModelParams)

  /////////////////////////////////////////////////////////////////////
  // Step 3: Get background passages for the positive and negative data
  /////////////////////////////////////////////////////////////////////

  val positiveBackgroundParams: JValue = ScienceCorpora.buscElasticSearchIndex(10) merge
    (("sentences" -> sentenceSelectorParams): JValue)

  val negativeBackgroundParams: JValue = ScienceCorpora.buscElasticSearchIndex(10) merge
    ("sentences" -> corruptedSentenceSelectorParams) ~
    ("remove query near duplicates" -> true)

}

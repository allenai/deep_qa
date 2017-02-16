package org.allenai.deep_qa.experiments.datasets

import org.json4s._
import org.json4s.JsonDSL._

/**
 * This object contains a bunch of JValue specifications for the
 * SciQ dataset's data files.
 */
object SciQDatasets {

  def sciQFile(sciQDir: String, split: String, version: String="1.0"): JValue = {
    val outputDirectory = sciQDir + "processed/"
    val inputFile = sciQDir + s"v${version}/"+ s"${split}-v${version}.json"
    val outputFiles = Seq(outputDirectory + s"${split}.tsv")
    ("sentence producer type" -> "dataset reader") ~
      ("reader" -> "sciq") ~
      ("create sentence indices" -> true) ~
      ("input file" -> inputFile) ~
      ("output files" -> outputFiles)
  }

  def sciQDataset(sciQDir: String, split: String, version: String="1.0"): JValue = {
    val file = sciQFile(sciQDir, split, version)
    ("data files" -> List(file))
  }

  // TODO(matt/nelson): It'd probably be a good idea to
  // have a DatasetUtil object, or something similar, that groups all
  // of the makeFile and makeDataset methods.
  def makeMcFile(sentence_params: JValue): JValue = {
    ("sentences" -> sentence_params) ~
      ("sentence producer type" -> "drop first column")
  }

  def makePassageBackgroundFile(
    sentences: JValue,
    sentenceFormat: String,
    corpus: JValue
  ) : JValue = {
    ("sentence producer type" -> "background searcher") ~
      ("searcher" -> corpus) ~
      ("sentences" -> sentences) ~
      ("sentence format" -> sentenceFormat)
  }

  val baseDir = "/efs/data/dlfa/sciq/"

  // Train files
  val trainFile = sciQFile(baseDir, "train")
  val train = sciQDataset(baseDir, "train")
  val mcTrainFile = makeMcFile(trainFile)
  val mcTrainBuscBackgroundFile = makePassageBackgroundFile(
    mcTrainFile,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )
  val readingComprehensionTrainWithBuscBackgroundFile: JValue =
    ("sentence producer type" -> "qa and background to rc") ~
      ("sentences" -> mcTrainFile) ~
      ("background" -> mcTrainBuscBackgroundFile)

  val mcTrainLuceneBackgroundFile = makePassageBackgroundFile(
    mcTrainFile,
    "question and answer",
    ScienceCorpora.aristoDefaultElasticSearchIndex(3)
  )
  val readingComprehensionTrainWithLuceneBackgroundFile: JValue =
    ("sentence producer type" -> "qa and background to rc") ~
      ("sentences" -> mcTrainFile) ~
      ("background" -> mcTrainLuceneBackgroundFile)

  val mcTrainWithBuscBackground: JValue = ScienceDatasets.makeBackgroundDataset(
    mcTrainFile,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )

  // Dev Files
  val devFile = sciQFile(baseDir, "dev")
  val dev = sciQDataset(baseDir, "dev")
  val mcDevFile = makeMcFile(devFile)
  val mcDevBuscBackgroundFile = makePassageBackgroundFile(
    mcDevFile,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )
  val readingComprehensionDevWithBuscBackgroundFile: JValue =
    ("sentence producer type" -> "qa and background to rc") ~
      ("sentences" -> mcDevFile) ~
      ("background" -> mcDevBuscBackgroundFile)

  val mcDevLuceneBackgroundFile = makePassageBackgroundFile(
    mcDevFile,
    "question and answer",
    ScienceCorpora.aristoDefaultElasticSearchIndex(3)
  )
  val readingComprehensionDevWithLuceneBackgroundFile: JValue =
    ("sentence producer type" -> "qa and background to rc") ~
      ("sentences" -> mcDevFile) ~
      ("background" -> mcDevLuceneBackgroundFile)

  val mcDevWithBuscBackground: JValue = ScienceDatasets.makeBackgroundDataset(
    mcDevFile,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )
  // Test Files
  val testFile = sciQFile(baseDir, "test")
  val test = sciQDataset(baseDir, "test")
  val mcTestFile = makeMcFile(testFile)
  val mcTestBuscBackgroundFile = makePassageBackgroundFile(
    mcTestFile,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )
  val readingComprehensionTestWithBuscBackgroundFile: JValue =
    ("sentence producer type" -> "qa and background to rc") ~
      ("sentences" -> mcTestFile) ~
      ("background" -> mcTestBuscBackgroundFile)

  val mcTestLuceneBackgroundFile = makePassageBackgroundFile(
    mcTestFile,
    "question and answer",
    ScienceCorpora.aristoDefaultElasticSearchIndex(3)
  )
  val readingComprehensionTestWithLuceneBackgroundFile: JValue =
    ("sentence producer type" -> "qa and background to rc") ~
      ("sentences" -> mcTestFile) ~
      ("background" -> mcTestLuceneBackgroundFile)

  val mcTestWithBuscBackground: JValue = ScienceDatasets.makeBackgroundDataset(
    mcTestFile,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )

  // Turn SciQ reading comprehension train file with BUSC background to a dataset.
  val sciQTrainDataset: JValue =
    ("dataset type" -> "from sentence producers") ~
      ("data files" -> Seq(SciQDatasets.readingComprehensionTrainWithBuscBackgroundFile))

}

object SciQDirectAnswer {
  def sciQFile(sciQDir: String, split: String, version: String="1.0"): JValue = {
    val outputDirectory = sciQDir + "processed/"
    val inputFile = sciQDir + s"${split}-v${version}.json"
    val outputFiles = Seq(outputDirectory + s"${split}.tsv")
    ("sentence producer type" -> "dataset reader") ~
    ("reader" -> "squad") ~
    ("input file" -> inputFile) ~
    ("output files" -> outputFiles)
  }

  def sciQDataset(sciQDir: String, split: String, version: String="1.0"): JValue = {
    val file = sciQFile(sciQDir, split, version)
    ("data files" -> List(file))
  }

  val baseDir = "/efs/data/dlfa/sciq_da/"

  val trainFile = sciQFile(baseDir, "train")
  val trainDataset = sciQDataset(baseDir, "train")
  val devFile = sciQFile(baseDir, "dev")
  val devDataset = sciQDataset(baseDir, "dev")
  val testFile = sciQFile(baseDir, "test")
  val testDataset = sciQDataset(baseDir, "test")
}

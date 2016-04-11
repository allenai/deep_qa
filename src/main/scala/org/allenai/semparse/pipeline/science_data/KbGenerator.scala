package org.allenai.semparse.pipeline.science_data

import org.json4s._

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import collection.mutable

/**
 * This step in the pipeline takes as input a training data file, and produces as output a relation
 * file, containing triples for all entities and relations seen enough times in the training data.
 */
class KbGenerator(
  params: JValue,
  fileUtil: FileUtil
) extends Step(Some(params), fileUtil) {
  override val name = "KB Generator"

  val validParams = Seq("min np count", "min relation count", "sentences")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val sentenceProcessor = new ScienceSentenceProcessor(params \ "sentences", fileUtil)
  val minNpCount = JsonHelper.extractWithDefault(params, "min np count", 25)
  val minRelationCount = JsonHelper.extractWithDefault(params, "min relation count", 100)

  val dataName = sentenceProcessor.dataName
  val trainingDataFile = sentenceProcessor.outputFile
  val tripleFile = s"data/science/$dataName/triples.tsv"

  override def inputs = Set(
    (trainingDataFile, Some(sentenceProcessor))
  )
  override val outputs = Set(tripleFile)
  override val paramFile = tripleFile.replace(".tsv", "_params.json")
  override val inProgressFile = tripleFile.replace(".tsv", "_in_progress")

  override def _runStep() {
    val npCounts = new mutable.HashMap[String, Int].withDefaultValue(0)
    val relationCounts = new mutable.HashMap[String, Int].withDefaultValue(0)
    val seenTriples = new mutable.HashSet[(String, String, String)]
    println("Iterating over the file")
    for (line <- fileUtil.getLineIterator(trainingDataFile)) {
      val fields = line.split("\t")
      val names = fields(1)
      val catWord = fields(2)
      val relWord = fields(3)
      if (catWord != "") {
        val subj = names.drop(1).dropRight(1)
        val obj = catWord
        val relation = "MODIFIER"
        npCounts.update(subj, npCounts(subj) + 1)
        relationCounts.update(catWord, relationCounts(catWord) + 1)
        val triple = (subj, relation, obj)
        seenTriples += triple
      } else {
        val nameFields = names.split("\" \"")
        val subj = nameFields(0).drop(1)
        val obj = nameFields(1).dropRight(1)
        val relation = relWord
        npCounts.update(subj, npCounts(subj) + 1)
        npCounts.update(obj, npCounts(obj) + 1)
        relationCounts.update(relation, relationCounts(relation) + 1)
        val triple = (subj, relation, obj)
        seenTriples += triple
      }
    }
    println("Filtering the seen triples")
    val keptTriples = seenTriples.par.filter(triple => {
      if (triple._2 == "MODIFIER") {
        npCounts(triple._1) >= minNpCount && relationCounts(triple._3) > minRelationCount
      } else {
        npCounts(triple._1) >= minNpCount && npCounts(triple._3) >= minNpCount &&
          relationCounts(triple._2) > minRelationCount
      }
    })
    val outputLines = keptTriples.map(t => s"${t._1}\t${t._3}\t${t._2}").seq.toSeq.sorted
    fileUtil.writeLinesToFile(tripleFile, outputLines)
  }
}

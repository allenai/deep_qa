package org.allenai.dlfa.pipeline

import org.json4s._
import org.json4s.JsonDSL._

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import edu.cmu.ml.rtw.pra.experiments.Outputter
import edu.cmu.ml.rtw.pra.graphs.GraphCreator

// This is mostly just a class to create a set of params and hand off to the GraphCreator in the
// PRA code.  Because of this, we don't pass our params to the base Step object.
class KbGraphCreator(
  params: JValue,
  fileUtil: FileUtil
) extends Step(None, fileUtil) {
  implicit val formats = DefaultFormats

  override val name = "KB Graph Creator"

  val validParams = Seq("graph name", "corpus triples", "relation sets")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val corpusRelationSet: Option[(JValue, (String, Option[Step]))] = (params \ "corpus triples") match {
    case JNothing => None
    case jval => {
      /*  TODO(matt): Fix this, if we ever need to use this again.  The KbGenerator went to a
       *  different repository.
      import org.allenai.semparse.pipeline.jklol.KbGenerator

      val kbGenerator = new KbGenerator(jval, fileUtil)
      val paramsForGraph: JValue = ("is kb" -> true) ~ ("relation file" -> kbGenerator.tripleFile)
      Some((paramsForGraph, (kbGenerator.tripleFile, Some(kbGenerator))))
      */
      None
    }
  }
  val suppliedRelationSetFiles = JsonHelper.extractWithDefault(params, "relation sets", Seq[String]())
  val suppliedRelationSets: Seq[(JValue, (String, Option[Step]))] = {
    suppliedRelationSetFiles.map(file => {
      val paramsForGraph: JValue = ("is kb" -> true) ~ ("relation file" -> file)
      (paramsForGraph, (file, None))
    })
  }
  val relationSets = corpusRelationSet.toSeq ++ suppliedRelationSets
  val graphName = (params \ "graph name").extract[String]
  val baseGraphDir = "data/science/graphs"
  val outputter = Outputter.justLogger

  val graphCreatorParams: JValue =
    ("name" -> graphName) ~
    ("relation sets" -> relationSets.map(_._1))

  val graphCreatorInputs: Set[(String, Option[Step])] = relationSets.map(_._2).toSet

  val graphCreator = new GraphCreator(baseGraphDir, graphCreatorParams, outputter, fileUtil) {
    override val inputs = graphCreatorInputs
  }

  val graphDir = graphCreator.outdir
  override val inputs: Set[(String, Option[Step])] = Set((s"$baseGraphDir/$graphName/edges.dat", Some(graphCreator)))
  override val outputs = graphCreator.outputs
  override val inProgressFile = s"$baseGraphDir/$graphName/kb_graph_creator_in_progress"

  override def _runStep() {
    // This is just a no-op, because the graphCreator we depend on did all of the work.
  }
}

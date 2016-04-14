package org.allenai.semparse.pipeline.science_data

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

  // TODO(matt): allow other relation sets.
  val validParams = Seq("graph name", "corpus triples")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val kbGenerator = new KbGenerator(params \ "corpus triples", fileUtil)
  val graphName = (params \ "graph name").extract[String]
  val baseGraphDir = "data/science/graphs"
  val outputter = Outputter.justLogger

  val graphCreatorParams: JValue =
    ("name" -> graphName) ~
    ("relation sets" -> List(("is kb" -> true) ~ ("relation file" -> kbGenerator.tripleFile)))

  val graphCreatorInputs: Set[(String, Option[Step])] = Set((kbGenerator.tripleFile, Some(kbGenerator)))

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

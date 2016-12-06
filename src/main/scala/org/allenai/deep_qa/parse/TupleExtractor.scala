package org.allenai.deep_qa.parse

import com.mattg.util.JsonHelper
import edu.knowitall.openie.OpenIE
import org.json4s._

abstract class TupleExtractor(params: JValue) {
  val baseParams = Seq("type", "max argument characters", "max sentence characters")

  val maxSentenceCharacters = JsonHelper.extractAsOption[Int](params, "max sentence characters")
  val maxArgumentCharacters = JsonHelper.extractAsOption[Int](params, "max argument characters")

  def extractTuples(sentence: String): Seq[Tuple] = {
    if (maxSentenceCharacters.map(sentence.length > _).getOrElse(false)) {
      Seq()
    } else {
      extract(sentence)
    }
  }

  protected def extract(sentence: String): Seq[Tuple]
}

object TupleExtractor {
  def create(params: JValue): TupleExtractor = {
    (params \ "type") match {
      case JString("open ie") => new OpenIeExtractor(params)
      case _ => throw new IllegalStateException("unrecognized tuple extractor parameters")
    }
  }
}

class OpenIeExtractor(params: JValue) extends TupleExtractor(params) {
  val openIE = new OpenIE()
  override protected def extract(sentence: String): Seq[Tuple] = {
    // Tushar says that the OpenIE code is not thread-safe, and will drop extractions if hit in
    // parallel.
    val extractions = (openIE synchronized {
      try {
        Some(openIE.extract(sentence))
      } catch {
        case _: Exception => None
      }
    }).getOrElse(Seq())
    extractions.flatMap(extraction => {
      val subject = extraction.extraction.arg1.text
      val predicate = extraction.extraction.rel.text
      val objects = extraction.extraction.arg2s.map(_.text)
      val subjectLengthOk = maxArgumentCharacters.map(subject.length < _).getOrElse(true)
      val predicateLengthOk = maxArgumentCharacters.map(predicate.length < _).getOrElse(true)
      val objectLengthsOk = maxArgumentCharacters.map(max => objects.forall(_.length < max)).getOrElse(true)
      if (subjectLengthOk && predicateLengthOk && objectLengthsOk) {
        Some(Tuple(subject, predicate, objects))
      } else {
        None
      }
    })
  }
}

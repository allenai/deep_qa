package org.allenai.deep_qa.data

/**
 * A single training / testing example.
 */
trait Instance {
  val label: Option[Any]

  /**
   * Converts the instance into a sequence of strings.  This is a Seq[String] instead of just a
   * String because some instance types (like BackgroundInstances) get written to multiple files,
   * and we need to separate them.
   */
  def asStrings(): Seq[String]
}

/**
 * An Instance that has a single true/false statement.
 */
case class TrueFalseInstance(
  statement: String,
  override val label: Option[Boolean]
) extends Instance {
  def asStrings(): Seq[String] = {
    label match {
      case Some(true) => Seq(s"$statement\t1")
      case Some(false) => Seq(s"$statement\t0")
      case None => Seq(s"$statement")
    }
  }
}

/**
 * An Instance that combines multiple true/false instances, where exactly one of them has label
 * true.  The label in this Instance is the index to the one whose label is true.
 */
case class MultipleTrueFalseInstance[T <: Instance](
  instances: Seq[Instance],
  override val label: Option[Int]
) extends Instance {

  /**
   * Here we return a single multiline string, one line for each statement.
   */
  def asStrings(): Seq[String] = {
    // TODO(matt): this needs to return a Seq[Seq[String]], to do things correctly...
    Seq()
  }
}

/**
 * An Instance that has question text and several answer options.
 */
case class QuestionAnswerInstance(
  question: String,
  answers: Seq[String],
  override val label: Option[Int]
) extends Instance {
  def asStrings(): Seq[String] = {
    val answerString = answers.mkString("###")
    label match {
      case Some(l) => Seq(s"$question\t$answerString\t$l")
      case None => Seq(s"$question\t$answerString")
    }
  }
}

/**
 * An Instance that wraps another Instance and adds background information.
 */
case class BackgroundInstance[T <: Instance](
  containedInstance: T,
  background: Seq[String]
) extends Instance {
  def asStrings(): Seq[String] = {
    val backgroundString = background.mkString("\t")
    containedInstance.asStrings() ++ Seq(backgroundString)
  }

  override val label = containedInstance.label
}

/**
 * An Instance created from the Stanford Natural Language Inference corpus.
 */
case class SnliInstance(
  text: String,
  hypothesis: String,
  override val label: Option[String]
) extends Instance {
  def asStrings(): Seq[String] = {
    label match {
      case Some(l) => Seq(s"$text\t$hypothesis\t$l")
      case None => Seq(s"$text\t$hypothesis")
    }
  }
}

/**
  * An Instance which can have multiple correct answers from a choice several answer options.
  */
case class MultipleCorrectQAInstance(
  question: String,
  answers: Seq[String],
  override val label: Option[Seq[Int]]
) extends Instance {
  def asStrings(): Seq[String] = {
    val answerString = answers.mkString("###")
    label match {
        case Some(l) => Seq(s"$question\t$answerString\t${l.mkString(",")}")
        case None => Seq(s"$question\t$answerString")
      }
  }
}



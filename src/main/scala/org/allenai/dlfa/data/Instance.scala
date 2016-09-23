package org.allenai.dlfa.data

/**
 * A single training / testing example.
 */
trait Instance {
  /**
   * Converts the instance into a sequence of strings.  This is a Seq[String] instead of just a
   * String because some instance types (like BackgroundInstances) get written to multiple files,
   * and we need to separate them.
   */
  def asStrings(): Seq[String]
}

/**
 * An Instance that has question text and several answer options.
 */
case class QuestionAnswerInstance(
  question: String,
  answers: Seq[String],
  label: Int
) extends Instance {
  def asStrings(): Seq[String] = {
    val answerString = answers.mkString("###")
    Seq(s"$question\t$answerString\t$label")
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
}

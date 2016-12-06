package org.allenai.deep_qa.parse

case class Tuple(subject: String, predicate: String, objects: Seq[String]) {
  def asString(separator: String="\t") = {
    val fields = Seq(subject, predicate) ++ objects
    fields.mkString(separator)
  }
}

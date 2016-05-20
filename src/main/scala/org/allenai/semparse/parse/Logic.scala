package org.allenai.semparse.parse

sealed trait Logic {
  def toLisp(): String
}

case class Atom(symbol: String) extends Logic {
  override def toString() = symbol
  override def toLisp() = "\"" + symbol + "\""
}

case class Predicate(predicate: String, arguments: Seq[Logic]) extends Logic {
  override def toString(): String = {
    val argString = arguments.mkString(", ")
    s"$predicate(${arguments.mkString(", ")})"
  }

  def toLisp(): String = {
    arguments.size match {
      case 1 => "((word-cat \"" + predicate + "\") " + arguments(0).toLisp + ")"
      case 2 => "((word-rel \"" + predicate + "\") " + arguments(0).toLisp + " " + arguments(1).toLisp + ")"
      case _ => throw new IllegalStateException("can't make lisp representation for predicate with more than 2 args")
    }
  }
}

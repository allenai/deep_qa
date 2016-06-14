package org.allenai.semparse.parse

sealed trait Logic {
  // If there's redundancy in how the logic is constructed, remove it.  Currently, this only does
  // anything for Conjunctions, and there it recursively removes all unnecessarily-nested
  // conjunctions.
  def flatten(): Logic
  def toLisp(): String
  def toJson(): String
}

case class Atom(symbol: String) extends Logic {
  override def flatten() = this
  override def toString() = symbol
  override def toLisp() = "\"" + symbol + "\""
  override def toJson() = "\"" + symbol + "\""
}

case class Predicate(predicate: String, arguments: Seq[Logic]) extends Logic {
  override def flatten() = {
    Predicate(predicate, arguments.map(_.flatten))
  }
  override def toString(): String = {
    val argString = arguments.mkString(", ")
    s"$predicate($argString)"
  }

  override def toJson(): String = {
    val argString = arguments.mkString(", ")
    "[\"" + predicate + "\"," + arguments.map(_.toJson).mkString(",") + "]"
  }

  def toLisp(): String = {
    arguments.size match {
      case 1 => "((word-cat \"" + predicate + "\") " + arguments(0).toLisp + ")"
      case 2 => "((word-rel \"" + predicate + "\") " + arguments(0).toLisp + " " + arguments(1).toLisp + ")"
      case _ => throw new IllegalStateException("can't make lisp representation for predicate with more than 2 args")
    }
  }
}

case class Conjunction(arguments: Set[Logic]) extends Logic {
  override def flatten(): Logic = {
    if (arguments.size == 1) {
      arguments.head.flatten
    } else {
      Conjunction(getFlattenedArguments())
    }
  }

  override def toString() = s"and(${arguments.mkString(", ")})"
  override def toLisp(): String = {
    "(and " + arguments.map(_.toLisp).mkString(" ") + ")"
  }
  override def toJson(): String = {
    "[" + arguments.map(_.toJson).mkString(", ") + "]"
  }

  private def getFlattenedArguments(): Set[Logic] = {
    arguments.flatMap(_ match {
      case c: Conjunction => {
        c.getFlattenedArguments()
      }
      case other => Set(other.flatten())
    })
  }
}

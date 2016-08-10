package org.allenai.dlfa.parse

import scala.collection.mutable

sealed trait Logic {
  // If there's redundancy in how the logic is constructed, remove it.  Currently, this only does
  // anything for Conjunctions, and there it recursively removes all unnecessarily-nested
  // conjunctions.
  def flatten(): Logic
  def toLisp(): String
  def toJson(): String
}

object Logic {
  def fromString(line: String): Logic = {
    // Currently, our logic statements will always start with a text token, which is either an atom
    // or a predicate.  The parsing approach we'll take is to first get the token, then decide if
    // we need to parse arguments.  If we do, we'll recurse on parsing the arguments using an
    // auxiliary recursive method.
    getPredicateAndArguments(0, line)._1
  }

  def getPredicateAndArguments(startIndex: Int, string: String): (Logic, Int) = {
    var token = ""
    var index = startIndex
    var arguments = new mutable.ArrayBuffer[Logic]
    while (index < string.length) {
      if (string(index) == '(') {
        val (argument, consumedUntil) = getPredicateAndArguments(index + 1, string)
        arguments += argument
        index = consumedUntil
      } else if (string(index) == ',') {
        return (combinePredicateAndArguments(token, arguments), index)
      } else if (string(index) == ')') {
        return (combinePredicateAndArguments(token, arguments), index)
      } else if (string(index) == ' ') {
        val (argument, consumedUntil) = getPredicateAndArguments(index + 1, string)
        arguments += argument
        index = consumedUntil
      } else {
        token += string(index)
      }
      index += 1
    }
    return (combinePredicateAndArguments(token, arguments), index)
  }

  def combinePredicateAndArguments(token: String, arguments: Seq[Logic]): Logic = {
    if (arguments.size == 0) {
      Atom(token)
    } else {
      if (token == "and") {
        Conjunction(arguments.toSet)
      } else {
        Predicate(token, arguments)
      }
    }
  }
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

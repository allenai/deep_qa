package org.allenai.semparse.ccg

import java.io.File
import java.util.Optional

import edu.uw.easysrl.dependencies.Coindexation
import edu.uw.easysrl.semantics._
import edu.uw.easysrl.semantics.lexicon._
import edu.uw.easysrl.syntax.grammar.Category
import edu.uw.easysrl.syntax.grammar.SyntaxTreeNode
import edu.uw.easysrl.syntax.parser.SRLParser.CCGandSRLparse

object ScienceQuestionLexicon {
  def getDefault(lexiconFile: File) = {
    println("Loading lexicon")
    lazy val manualLexicon: Option[Lexicon] = if (lexiconFile.exists()) Some(new ManualLexicon(lexiconFile)) else None
    lazy val lexica = Seq(
      new DeterminerLexicon,
      new NounLexicon,
      new AdjectiveLexicon,
      new VerbLexicon,
      new ModalLexicon,
      new CopulaLexicon,
      new NumbersLexicon,
      new DefaultLexicon
    ) ++ manualLexicon.toList
    new CompositeLexicon(lexica)
  }
}

// I basically have to duplicate the CompositeLexicon that exists in the EasySRL code, because Mike
// made the constructor private...
class CompositeLexicon(lexica: Seq[Lexicon]) extends Lexicon {
  override def getEntry(
    word: String,
    pos: String,
    category: Category,
    coindexation: Coindexation,
    parse: Optional[CCGandSRLparse],
    wordIndex: Int
  ) = {
    lexica.view.map(lexicon => {
      lexicon.getEntry(word, pos, category, coindexation, parse, wordIndex)
    }).filter(_ != null).headOption match {
      case None => null
      case Some(logic) => {
        println("\nComposite matched something:")
        println(s"word: $word")
        println(s"pos: $pos")
        println(s"category: $category")
        println(s"logic: $logic\n")
        logic
      }
    }
  }

  override def isMultiWordExpression(node: SyntaxTreeNode) = {
    lexica.exists(_.isMultiWordExpression(node))
  }
}

class VerbLexicon extends Lexicon {
  override def getEntry(
    word: String,
    pos: String,
    category: Category,
    coindexation: Coindexation,
    parse: Optional[CCGandSRLparse],
    wordIndex: Int
  ) = {
    println(s"Trying verb lexicon; number of arguments for category ${category}: ${category.getNumberOfArguments}")
    if (category.getNumberOfArguments == 2) {
      println(s"Two arguments, with categories ${category.getArgument(0)} and ${category.getArgument(1)}")
      println(s"${category.getRight}")
      println(s"${category.getLeft}")
      println(s"${category.getLeft.getRight}")
      if (category.getRight().equals(Category.NP) && category.getLeft().getRight().equals(Category.NP)) {
        // (S\NP)/NP
        val arg1 = new Variable(SemanticType.E)
        val arg2 = new Variable(SemanticType.E)
        new LambdaExpression(
          new AtomicSentence(word, arg1, arg2),
          arg2,
          arg1
        )
      } else {
        null
      }
    } else {
      null
    }
  }

  override def isMultiWordExpression(node: SyntaxTreeNode) = {
    // TODO(matt): I might want to change this...
    false
  }
}

class ModalLexicon extends Lexicon {
  override def getEntry(
    word: String,
    pos: String,
    category: Category,
    coindexation: Coindexation,
    parse: Optional[CCGandSRLparse],
    wordIndex: Int
  ) = {
    if (pos == "MD") {
      val variable = new Variable(SemanticType.E)
      new LambdaExpression(variable, variable)
    } else {
      null
    }
  }

  override def isMultiWordExpression(node: SyntaxTreeNode) = {
    // TODO(matt): I might want to change this...
    false
  }
}

class NounLexicon extends Lexicon {
  override def getEntry(
    word: String,
    pos: String,
    category: Category,
    coindexation: Coindexation,
    parse: Optional[CCGandSRLparse],
    wordIndex: Int
  ) = {
    if (category == Category.N && pos.startsWith("NN")) {
      new Constant(getLemma(word, pos, parse, wordIndex), SemanticType.E)
    } else {
      null
    }
  }

  override def isMultiWordExpression(node: SyntaxTreeNode) = {
    // TODO(matt): I might want to change this...
    false
  }
}

class AdjectiveLexicon extends Lexicon {
  override def getEntry(
    word: String,
    pos: String,
    category: Category,
    coindexation: Coindexation,
    parse: Optional[CCGandSRLparse],
    wordIndex: Int
  ) = {
    if (category == Category.valueOf("N/N")) {
      val noun = new Variable(SemanticType.E)
      new LambdaExpression(new AtomicSentence(word, noun), noun)
    } else {
      null
    }
  }

  override def isMultiWordExpression(node: SyntaxTreeNode) = {
    // TODO(matt): I might want to change this...
    false
  }
}

class DeterminerLexicon extends Lexicon {
  override def getEntry(
    word: String,
    pos: String,
    category: Category,
    coindexation: Coindexation,
    parse: Optional[CCGandSRLparse],
    wordIndex: Int
  ) = {
    if (category == Category.valueOf("NP/N") || category == Category.DETERMINER) {
      val variable = new Variable(SemanticType.E)
      new LambdaExpression(variable, variable)
    } else {
      null
    }
  }

  override def isMultiWordExpression(node: SyntaxTreeNode) = {
    // TODO(matt): I might want to change this...
    false
  }
}

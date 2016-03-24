package org.allenai.semparse.ccg

import org.scalatest._

import java.io.File
import java.util.Optional

import edu.uw.easysrl.dependencies.Coindexation
import edu.uw.easysrl.semantics.Logic
import edu.uw.easysrl.semantics.LogicParser
import edu.uw.easysrl.semantics.lexicon.Lexicon
import edu.uw.easysrl.syntax.grammar.Category
import edu.uw.easysrl.syntax.grammar.Combinator.RuleType
import edu.uw.easysrl.syntax.grammar.SyntaxTreeNode
import edu.uw.easysrl.syntax.grammar.SyntaxTreeNode._
import edu.uw.easysrl.syntax.parser.AbstractParser.UnaryRule
import edu.uw.easysrl.syntax.parser.SRLParser.CCGandSRLparse

class CompositeLexiconSpec extends FlatSpecLike with Matchers {
  val logic1 = LogicParser.fromString("#y.y", Category.valueOf("N"))
  val logic2 = LogicParser.fromString("#x.x", Category.valueOf("S"))
  val logic3 = LogicParser.fromString("#z.z", Category.valueOf("N"))
  class FakeLexicon(_word: Option[String], _category: Category, logic: Logic) extends Lexicon {
    override def getEntry(
      word: String,
      pos: String,
      category: Category,
      coindexation: Coindexation,
      parse: Optional[CCGandSRLparse],
      wordIndex: Int
    ) = {
      if (_category.matches(category)) {
        _word match {
          case None => logic
          case Some(w) => if (word == w) logic else null
        }
      } else {
        null
      }
    }
    override def isMultiWordExpression(node: SyntaxTreeNode) = false
  }
  val lexicon1 = new FakeLexicon(Some("one"), Category.valueOf("N"), logic1)
  val lexicon2 = new FakeLexicon(None, Category.valueOf("S"), logic2)
  val lexicon3 = new FakeLexicon(None, Category.valueOf("N"), logic3)

  "getEntry" should "ask each lexicon in turn and return the first match" in {
    val lexicon = new CompositeLexicon(Seq(lexicon1, lexicon2, lexicon3))
    lexicon.getEntry("one", "NN", Category.valueOf("N"), null, null, 0) should be(logic1)
    lexicon.getEntry("two", "NN", Category.valueOf("N"), null, null, 0) should be(logic3)
    lexicon.getEntry("", "VB", Category.valueOf("S"), null, null, 0) should be(logic2)

    val differentOrder = new CompositeLexicon(Seq(lexicon3, lexicon2, lexicon1))
    differentOrder.getEntry("one", "NN", Category.valueOf("N"), null, null, 0) should be(logic3)
    differentOrder.getEntry("two", "NN", Category.valueOf("N"), null, null, 0) should be(logic3)
    differentOrder.getEntry("", "VB", Category.valueOf("S"), null, null, 0) should be(logic2)
  }
}

class ScienceQuestionLexiconSpec extends FlatSpecLike with Matchers {

  val lexicon = ScienceQuestionLexicon.getDefault(new File("models/easysrl_sentences/lexicon"))

  val parse = new SyntaxTreeNodeBinary(
    Category.Sdcl,
    new SyntaxTreeNodeBinary(
      Category.NP,
      new SyntaxTreeNodeLeaf("A", "DT", null, Category.valueOf("NP/N"), 0),
      new SyntaxTreeNodeBinary(
        Category.N,
        new SyntaxTreeNodeLeaf("human", "JJ", null, Category.valueOf("N/N"), 1),
        new SyntaxTreeNodeLeaf("offspring", "NN", null, Category.N, 2),
        RuleType.FA,
        false,
        null,
        null),
      RuleType.FA,
      false,
      null,
      null),
    new SyntaxTreeNodeBinary(
      Category.valueOf("S[dcl]\\NP"),
      new SyntaxTreeNodeLeaf("can", "MD", null, Category.valueOf("(S[dcl]\\NP)/(S[b]\\NP)"), 3),
      new SyntaxTreeNodeBinary(
        Category.valueOf("S[b]\\NP"),
        new SyntaxTreeNodeLeaf("inherit", "VB", null, Category.valueOf("(S[b]\\NP)/NP"), 4),
        new SyntaxTreeNodeUnary(
          Category.valueOf("NP"),
          new SyntaxTreeNodeBinary(
            Category.valueOf("N"),
            new SyntaxTreeNodeLeaf("blue", "JJ", null, Category.valueOf("N/N"), 5),
            new SyntaxTreeNodeLeaf("eyes", "NNS", null, Category.N, 6),
            RuleType.FA,
            false,
            null,
            null),
          null,
          new UnaryRule(0, "N_1", "NP_1", LogicParser.fromString("#p.sk(#x.p(x))", Category.N)),
          null),
        RuleType.FA,
        true,
        null,
        null),
      RuleType.FA,
      false,
      null,
      null),
    RuleType.BA,
    false,
    null,
    null)

  "getEntry" should "give a correct logical form for a simple sentence" in {
    val ccgAndSrl = new CCGandSRLparse(parse, new java.util.ArrayList(), null)
    val withSemantics = ccgAndSrl.addSemantics(lexicon)
    // #e.(inherit(e)&ARG0(sk(#x.(offspring(x)&human(x))),e)&ARG1(sk(#y.(eye(y)&blue(y))),e))
    println(withSemantics.getCcgParse.getSemantics.get)
  }
}

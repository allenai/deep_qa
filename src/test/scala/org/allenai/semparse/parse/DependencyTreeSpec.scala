package org.allenai.semparse.parse

import org.scalatest._

class DependencyTreeSpec extends FlatSpecLike with Matchers {
  val tree =
    DependencyTree(Token("eat", "VBP", "eat", 2), Seq(
      (DependencyTree(Token("People", "NNS", "people", 1), Seq()), "nsubj"),
      (DependencyTree(Token("food", "NN", "food", 4), Seq(
        (DependencyTree(Token("good", "JJ", "good", 3), Seq()), "amod"))), "dobj")))

  val np1 =
    DependencyTree(Token("material", "NN", "material", 4), Seq(
      (DependencyTree(Token("genetic", "JJ", "genetic", 3), Seq()), "amod"),
      (DependencyTree(Token("called", "VBN", "call", 5), Seq(
        (DependencyTree(Token("DNA", "NN", "dna", 6), Seq()), "dobj"))), "vmod")))

  "_yield" should "return all of the tokens in a subtree, sorted by index" in {
    tree._yield should be("People eat good food")
    tree.children(0)._1._yield should be("People")
    tree.children(1)._1._yield should be("good food")
    tree.children(1)._1.children(0)._1._yield should be("good")
  }

  "simplifications" should "simplify a verb" in {
    // I don't really anticipate using this, but we'll check that it works as expected in this
    // case, too.
    tree.simplifications should be(Set(
      tree,
      DependencyTree(Token("eat", "VBP", "eat", 2), Seq(
        (DependencyTree(Token("People", "NNS", "people", 1), Seq()), "nsubj"))),
      DependencyTree(Token("eat", "VBP", "eat", 2), Seq())
    ))
    np1.simplifications should be(Set(
      np1,
      DependencyTree(Token("material", "NN", "material", 4), Seq(
        (DependencyTree(Token("genetic", "JJ", "genetic", 3), Seq()), "amod"))),
      DependencyTree(Token("material", "NN", "material", 4), Seq())
    ))
  }
}

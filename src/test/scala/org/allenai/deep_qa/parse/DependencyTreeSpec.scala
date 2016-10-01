package org.allenai.dlfa.parse

import org.scalatest._

class DependencyTreeSpec extends FlatSpecLike with Matchers {
  val tree =
    DependencyTree(Token("eat", "VBP", "eat", 2), Seq(
      (DependencyTree(Token("People", "NNS", "people", 1), Seq()), "nsubj"),
      (DependencyTree(Token("food", "NN", "food", 4), Seq(
        (DependencyTree(Token("good", "JJ", "good", 3), Seq()), "amod"))), "dobj")))

  val passiveTree =
    DependencyTree(Token("eaten", "VBN", "eat", 3), Seq(
      (DependencyTree(Token("Food", "NN", "food", 1), Seq()), "nsubjpass"),
      (DependencyTree(Token("is", "VB", "be", 2), Seq()), "auxpass"),
      (DependencyTree(Token("people", "NNS", "people", 5), Seq()), "agent")))

  val np1 =
    DependencyTree(Token("material", "NN", "material", 4), Seq(
      (DependencyTree(Token("genetic", "JJ", "genetic", 3), Seq()), "amod"),
      (DependencyTree(Token("called", "VBN", "call", 5), Seq(
        (DependencyTree(Token("DNA", "NN", "dna", 6), Seq()), "dobj"))), "vmod")))

  val np2 =
    DependencyTree(Token("Most", "JJS", "most", 1), Seq(
      (DependencyTree(Token("water", "NNP", "water", 4), Seq(
        (DependencyTree(Token("Earth", "NNP", "Earth", 3), Seq()), "poss"))),"prep_of")))

  val np3 =
    DependencyTree(Token("Most", "JJS", "most", 1), Seq(
      (DependencyTree(Token("water", "NNP", "water", 4), Seq(
        (DependencyTree(Token("their", "PRP$", "they", 3), Seq()), "poss"))),"prep_of")))

  val treeWithConjunctions =
    DependencyTree(Token("cause", "VB", "cause", 6), Seq(
      (DependencyTree(Token("Water", "NN", "water", 1), Seq(
        (DependencyTree(Token("wind", "NN", "wind", 3), Seq()), "conj_and"),
        (DependencyTree(Token("animals", "NNS", "animal", 5), Seq()), "conj_and"))), "nsubj"),
      (DependencyTree(Token("wind", "NN", "wind", 3), Seq()), "nsubj"),
      (DependencyTree(Token("animals", "NNS", "animal", 5), Seq()), "nsubj"),
      (DependencyTree(Token("erosion", "NN", "erosion", 7), Seq()), "dobj")))

  "_yield" should "return all of the tokens in a subtree, sorted by index" in {
    tree._yield should be("People eat good food")
    tree.children(0)._1._yield should be("People")
    tree.children(1)._1._yield should be("good food")
    tree.children(1)._1.children(0)._1._yield should be("good")
  }

  it should "add back in prepositions and possessives" in {
    np2._yield should be("Most of Earth's water")
  }

  it should "not add 's when the possessive is a pronoun" in {
    np3._yield should be("Most of their water")
  }

  it should "add back in \"by\" in passive sentences with an agent" in {
    passiveTree._yield should be("Food is eaten by people")
  }

  it should "not repeat conjunctions" in {
    treeWithConjunctions._yield should be("Water, wind and animals cause erosion")
  }

  "lemmaYield" should "return the lemmas of all of the tokens in a subtree, sorted by index" in {
    tree.lemmaYield should be("people eat good food")
    tree.children(0)._1.lemmaYield should be("people")
    tree.children(1)._1.lemmaYield should be("good food")
    tree.children(1)._1.children(0)._1.lemmaYield should be("good")
  }

  it should "add back in prepositions and possessives" in {
    np2.lemmaYield should be("most of Earth's water")
  }

  it should "not repeat conjunctions" in {
    treeWithConjunctions.lemmaYield should be("water, wind and animal cause erosion")
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

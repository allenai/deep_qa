package org.allenai.semparse.parse

import org.scalatest._

class TreeTransformerSpec extends FlatSpecLike with Matchers {

  // These trees are tricky to write down like this...  Hopefully the structure is clear from this
  // code formatting.
  val sentenceTrees: Map[String, DependencyTree] = Map(
    "Cells contain genetic material called DNA." ->
      DependencyTree(Token("contain", "VBP", "contain", 2), Seq(
        (DependencyTree(Token("Cells", "NNS", "cell", 1), Seq()), "nsubj"),
        (
          DependencyTree(Token("material", "NN", "material", 4), Seq(
            (DependencyTree(Token("genetic", "JJ", "genetic", 3), Seq()), "amod"),
            (
              DependencyTree(Token("called", "VBN", "call", 5), Seq(
                (DependencyTree(Token("DNA", "NN", "dna", 6), Seq()), "dobj")
              )),
              "vmod"
            )
          )),
          "dobj"
        )
      )),
    "Most of Earth is covered by water." ->
      DependencyTree(Token("covered", "VBN", "cover", 5), Seq(
        (
          DependencyTree(Token("Most", "JJS", "most", 1), Seq(
            (DependencyTree(Token("Earth", "NNP", "Earth", 3), Seq()), "prep_of")
          )),
          "nsubjpass"
        ),
        (DependencyTree(Token("is", "VBZ", "be", 4), Seq()), "auxpass"),
        (DependencyTree(Token("water", "NN", "water", 7), Seq()), "agent")
      )),
    "Which gas is given off by plants?" ->
      DependencyTree(Token("given", "VBN", "give", 4), Seq(
        (
          DependencyTree(Token("gas", "NN", "gas", 2), Seq(
            (DependencyTree(Token("Which", "WDT", "which", 1), Seq()), "det")
          )),
          "nsubjpass"
        ),
        (DependencyTree(Token("is", "VBZ", "be", 3), Seq()), "auxpass"),
        (DependencyTree(Token("off", "RP", "off", 5), Seq()), "prt"),
        (DependencyTree(Token("plants", "NNS", "plant", 7), Seq()), "agent")
      )),
    "Which of these is an example of liquid water?" ->
      DependencyTree(Token("is", "VBZ", "be", 4), Seq(
        (
          DependencyTree(Token("Which", "WDT", "which", 1), Seq(
            (DependencyTree(Token("these", "DT", "these", 3), Seq()), "prep_of")
          )),
          "dep"
        ),
        (
          DependencyTree(Token("example", "NN", "example", 6), Seq(
            (DependencyTree(Token("an", "DT", "a", 5), Seq()), "det"),
            (
              DependencyTree(Token("water", "NN", "water", 9), Seq(
                (DependencyTree(Token("liquid", "JJ", "liquid", 8), Seq()), "amod")
              )),
              "prep_of"
            )
          )),
          "nsubj"
        )
      )),
    "The seeds of an oak come from the fruit." ->
      DependencyTree(Token("come", "VBN", "come", 6), Seq(
        (
          DependencyTree(Token("seeds", "NNS", "seed", 2), Seq(
            (DependencyTree(Token("The", "DT", "the", 1), Seq()), "det"),
            (
              DependencyTree(Token("oak", "NN", "oak", 5), Seq(
                (DependencyTree(Token("an", "DT", "a", 4), Seq()), "det")
              )),
              "prep_of"
            )
          )),
          "nsubj"
        ),
        (
          DependencyTree(Token("fruit", "NN", "fruit", 9), Seq(
            (DependencyTree(Token("the", "DT", "the", 8), Seq()), "det")
          )),
          "prep_from"
        )
      ))
  )

  "replaceChild" should "leave the rest of the tree intact, and replace one child" in {
    val tree =
      DependencyTree(Token("given", "VBN", "give", 4), Seq(
        (DependencyTree(Token("plants", "NNS", "plant", 7), Seq()), "nsubj"),
        (DependencyTree(Token("off", "RP", "off", 5), Seq()), "prt"),
        (DependencyTree(Token("gas", "NN", "gas", 2), Seq(
            (DependencyTree(Token("Which", "WDT", "which", 1), Seq()), "det"))), "dobj")))

    val childLabel = "prt"

    val newChild =
      DependencyTree(Token("Most", "JJS", "most", 1), Seq(
        (DependencyTree(Token("Earth", "NNP", "Earth", 2), Seq()), "prep_of")))

    val newLabel = "new label"

    val expectedTree =
      DependencyTree(Token("given", "VBN", "give", 4), Seq(
        (DependencyTree(Token("plants", "NNS", "plant", 7), Seq()), "nsubj"),
        (DependencyTree(Token("Most", "JJS", "most", 1), Seq(
          (DependencyTree(Token("Earth", "NNP", "Earth", 2), Seq()), "prep_of"))), newLabel),
        (DependencyTree(Token("gas", "NN", "gas", 2), Seq(
            (DependencyTree(Token("Which", "WDT", "which", 1), Seq()), "det"))), "dobj")))

    transformers.replaceChild(tree, childLabel, newChild, newLabel) should be(expectedTree)
  }

  "replaceTree" should "replace whole trees" in {
    val tree =
      DependencyTree(Token("given", "VBN", "give", 4), Seq(
        (DependencyTree(Token("replace", "VB", "replace", 0), Seq()), "nsubj"),
        (DependencyTree(Token("off", "RP", "off", 5), Seq()), "prt"),
        (DependencyTree(Token("gas", "NN", "gas", 2), Seq(
            (DependencyTree(Token("replace", "VB", "replace", 0), Seq()), "det"))), "dobj")))

    val toReplace = DependencyTree(Token("replace", "VB", "replace", 0), Seq())
    val replaceWith =
      DependencyTree(Token("replaced", "VBD", "replaced", 0), Seq(
          (DependencyTree(Token("ha!", "!!", "ha!", 0), Seq()), "funny")))

    val expectedTree =
      DependencyTree(Token("given", "VBN", "give", 4), Seq(
        (DependencyTree(Token("replaced", "VBD", "replaced", 0), Seq(
            (DependencyTree(Token("ha!", "!!", "ha!", 0), Seq()), "funny"))), "nsubj"),
        (DependencyTree(Token("off", "RP", "off", 5), Seq()), "prt"),
        (DependencyTree(Token("gas", "NN", "gas", 2), Seq(
            (DependencyTree(Token("replaced", "VBD", "replaced", 0), Seq(
                (DependencyTree(Token("ha!", "!!", "ha!", 0), Seq()), "funny"))), "det"))), "dobj")))

    transformers.replaceTree(tree, toReplace, replaceWith) should be(expectedTree)
  }

  "removeChild" should "leave the rest of the tree intact, and remove one child" in {
    val tree =
      DependencyTree(Token("given", "VBN", "give", 4), Seq(
        (DependencyTree(Token("plants", "NNS", "plant", 7), Seq()), "nsubj"),
        (DependencyTree(Token("off", "RP", "off", 5), Seq()), "prt"),
        (DependencyTree(Token("gas", "NN", "gas", 2), Seq(
            (DependencyTree(Token("Which", "WDT", "which", 1), Seq()), "det"))), "dobj")))

    val childLabel = "prt"

    val expectedTree =
      DependencyTree(Token("given", "VBN", "give", 4), Seq(
        (DependencyTree(Token("plants", "NNS", "plant", 7), Seq()), "nsubj"),
        (DependencyTree(Token("gas", "NN", "gas", 2), Seq(
            (DependencyTree(Token("Which", "WDT", "which", 1), Seq()), "det"))), "dobj")))

    transformers.removeChild(tree, childLabel) should be(expectedTree)
  }

  "removeTree" should "find matching trees, and remove them" in {
    val tree =
      DependencyTree(Token("given", "VBN", "give", 4), Seq(
        (DependencyTree(Token("remove", "VB", "remove", 0), Seq()), "nsubj"),
        (DependencyTree(Token("off", "RP", "off", 5), Seq()), "prt"),
        (DependencyTree(Token("gas", "NN", "gas", 2), Seq(
            (DependencyTree(Token("remove", "VB", "remove", 0), Seq()), "det"))), "dobj")))

    val toRemove = DependencyTree(Token("remove", "VB", "remove", 0), Seq())

    val expectedTree =
      DependencyTree(Token("given", "VBN", "give", 4), Seq(
        (DependencyTree(Token("off", "RP", "off", 5), Seq()), "prt"),
        (DependencyTree(Token("gas", "NN", "gas", 2), Seq()), "dobj")))

    transformers.removeTree(tree, toRemove) should be(expectedTree)
  }

  "findWhPhrase" should "find the correct wh-phrase" in {
    val tree1 = sentenceTrees("Which gas is given off by plants?")
    transformers.findWhPhrase(tree1) should be(Some(tree1.children(0)._1))
    val tree2 = sentenceTrees("Which of these is an example of liquid water?")
    transformers.findWhPhrase(tree2) should be(Some(tree2.children(0)._1))
  }

  "UndoPassivization" should "switch nsubjpass to dobj, and agent to nsubj" in {
    val tree1 = sentenceTrees("Most of Earth is covered by water.")
    transformers.UndoPassivization.transform(tree1) should be(
      DependencyTree(Token("covered", "VBN", "cover", 5), Seq(
        (DependencyTree(Token("water", "NN", "water", 7), Seq()), "nsubj"),
        (DependencyTree(Token("Most", "JJS", "most", 1), Seq(
            (DependencyTree(Token("Earth", "NNP", "Earth", 3), Seq()), "prep_of"))), "dobj")))
    )
    val tree2 = sentenceTrees("Which gas is given off by plants?")
    transformers.UndoPassivization.transform(tree2) should be(
      DependencyTree(Token("given", "VBN", "give", 4), Seq(
        (DependencyTree(Token("plants", "NNS", "plant", 7), Seq()), "nsubj"),
        (DependencyTree(Token("off", "RP", "off", 5), Seq()), "prt"),
        (DependencyTree(Token("gas", "NN", "gas", 2), Seq(
            (DependencyTree(Token("Which", "WDT", "which", 1), Seq()), "det"))), "dobj")))
    )
  }

  it should "leave trees alone when there is no passive" in {
    val tree = sentenceTrees("Which of these is an example of liquid water?")
    transformers.UndoPassivization.transform(tree) should be(tree)
  }

  "ReplaceWhPhrase" should "find the wh-phrase, then replace it with a given tree" in {
    val answerTree = DependencyTree(Token("answer", "NN", "answer", 0), Seq())
    val tree1 = sentenceTrees("Which gas is given off by plants?")
    new transformers.ReplaceWhPhrase(answerTree).transform(tree1) should be(
      DependencyTree(Token("given", "VBN", "give", 4), Seq(
        (answerTree, "nsubjpass"),
        (DependencyTree(Token("is", "VBZ", "be", 3), Seq()), "auxpass"),
        (DependencyTree(Token("off", "RP", "off", 5), Seq()), "prt"),
        (DependencyTree(Token("plants", "NNS", "plant", 7), Seq()), "agent")
      ))
    )
    val tree2 = sentenceTrees("Which of these is an example of liquid water?")
    new transformers.ReplaceWhPhrase(answerTree).transform(tree2) should be(
      DependencyTree(Token("is", "VBZ", "be", 4), Seq(
        (answerTree, "dep"),
        (DependencyTree(Token("example", "NN", "example", 6), Seq(
          (DependencyTree(Token("an", "DT", "a", 5), Seq()), "det"),
          (DependencyTree(Token("water", "NN", "water", 9), Seq(
            (DependencyTree(Token("liquid", "JJ", "liquid", 8), Seq()), "amod"))), "prep_of"))), "nsubj")))
    )
  }

  it should "do nothing on a tree with no wh-phrase" in {
    val answerTree = DependencyTree(Token("answer", "NN", "answer", 0), Seq())
    val tree = sentenceTrees("Most of Earth is covered by water.")
    new transformers.ReplaceWhPhrase(answerTree).transform(tree) should be(tree)
  }

  "RemoveDeterminers" should "remove determiners from the tree" in {
    val tree1 = sentenceTrees("Which gas is given off by plants?")
    transformers.RemoveDeterminers.transform(tree1) should be(
      DependencyTree(Token("given", "VBN", "give", 4), Seq(
        (DependencyTree(Token("gas", "NN", "gas", 2), Seq()), "nsubjpass"),
        (DependencyTree(Token("is", "VBZ", "be", 3), Seq()), "auxpass"),
        (DependencyTree(Token("off", "RP", "off", 5), Seq()), "prt"),
        (DependencyTree(Token("plants", "NNS", "plant", 7), Seq()), "agent"))))
    val tree2 = sentenceTrees("The seeds of an oak come from the fruit.")
    transformers.RemoveDeterminers.transform(tree2) should be(
      DependencyTree(Token("come", "VBN", "come", 6), Seq(
        (DependencyTree(Token("seeds", "NNS", "seed", 2), Seq(
          (DependencyTree(Token("oak", "NN", "oak", 5), Seq()), "prep_of"))), "nsubj"),
        (DependencyTree(Token("fruit", "NN", "fruit", 9), Seq()), "prep_from"))))
  }

  "CombineParticles" should "put particles into verb lemmas" in {
    val tree = sentenceTrees("Which gas is given off by plants?")
    transformers.CombineParticles.transform(tree) should be(
      DependencyTree(Token("given_off", "VBN", "give_off", 4), Seq(
        (DependencyTree(Token("gas", "NN", "gas", 2), Seq(
          (DependencyTree(Token("Which", "WDT", "which", 1), Seq()), "det"))), "nsubjpass"),
        (DependencyTree(Token("is", "VBZ", "be", 3), Seq()), "auxpass"),
        (DependencyTree(Token("plants", "NNS", "plant", 7), Seq()), "agent")
      )))
  }
}

package org.allenai.semparse.parse

import org.scalatest._

class TreeTransformerSpec extends FlatSpecLike with Matchers {

  // These trees are tricky to write down like this...  Hopefully the structure is clear from this
  // code formatting.
  val sentenceTrees: Map[String, DependencyTree] = Map(
    "Cells contain genetic material called DNA." ->
      DependencyTree(Token("contain", "VBP", "contain"), Seq(
        (DependencyTree(Token("Cells", "NNS", "cell"), Seq()), "nsubj"),
        (
          DependencyTree(Token("material", "NN", "material"), Seq(
            (DependencyTree(Token("genetic", "JJ", "genetic"), Seq()), "amod"),
            (
              DependencyTree(Token("called", "VBN", "call"), Seq(
                (DependencyTree(Token("DNA", "NN", "dna"), Seq()), "dobj")
              )),
              "vmod"
            )
          )),
          "dobj"
        )
      )),
    "Most of Earth is covered by water." ->
      DependencyTree(Token("covered", "VBN", "cover"), Seq(
        (
          DependencyTree(Token("Most", "JJS", "most"), Seq(
            (DependencyTree(Token("Earth", "NNP", "Earth"), Seq()), "prep_of")
          )),
          "nsubjpass"
        ),
        (DependencyTree(Token("is", "VBZ", "be"), Seq()), "auxpass"),
        (DependencyTree(Token("water", "NN", "water"), Seq()), "agent")
      )),
    "Which gas is given off by plants?" ->
      DependencyTree(Token("given", "VBN", "give"), Seq(
        (
          DependencyTree(Token("gas", "NN", "gas"), Seq(
            (DependencyTree(Token("Which", "WDT", "which"), Seq()), "det")
          )),
          "nsubjpass"
        ),
        (DependencyTree(Token("is", "VBZ", "be"), Seq()), "auxpass"),
        (DependencyTree(Token("off", "RP", "off"), Seq()), "prt"),
        (DependencyTree(Token("plants", "NNS", "plant"), Seq()), "agent")
      )),
    "Which of these is an example of liquid water?" ->
      DependencyTree(Token("is", "VBZ", "be"), Seq(
        (
          DependencyTree(Token("Which", "WDT", "which"), Seq(
            (DependencyTree(Token("these", "DT", "these"), Seq()), "prep_of")
          )),
          "dep"
        ),
        (
          DependencyTree(Token("example", "NN", "example"), Seq(
            (DependencyTree(Token("an", "DT", "a"), Seq()), "det"),
            (
              DependencyTree(Token("water", "NN", "water"), Seq(
                (DependencyTree(Token("liquid", "JJ", "liquid"), Seq()), "amod")
              )),
              "prep_of"
            )
          )),
          "nsubj"
        )
      ))
  )

  "replaceChild" should "leave the rest of the tree intact, and replace one child" in {
    val tree =
      DependencyTree(Token("given", "VBN", "give"), Seq(
        (DependencyTree(Token("plants", "NNS", "plant"), Seq()), "nsubj"),
        (DependencyTree(Token("off", "RP", "off"), Seq()), "prt"),
        (DependencyTree(Token("gas", "NN", "gas"), Seq(
            (DependencyTree(Token("Which", "WDT", "which"), Seq()), "det"))), "dobj")))

    val childLabel = "prt"

    val newChild =
      DependencyTree(Token("Most", "JJS", "most"), Seq(
        (DependencyTree(Token("Earth", "NNP", "Earth"), Seq()), "prep_of")))

    val newLabel = "new label"

    val expectedTree =
      DependencyTree(Token("given", "VBN", "give"), Seq(
        (DependencyTree(Token("plants", "NNS", "plant"), Seq()), "nsubj"),
        (DependencyTree(Token("Most", "JJS", "most"), Seq(
          (DependencyTree(Token("Earth", "NNP", "Earth"), Seq()), "prep_of"))), newLabel),
        (DependencyTree(Token("gas", "NN", "gas"), Seq(
            (DependencyTree(Token("Which", "WDT", "which"), Seq()), "det"))), "dobj")))

    transformers.replaceChild(tree, childLabel, newChild, newLabel) should be(expectedTree)
  }

  "replaceTree" should "replace whole trees" in {
    val tree =
      DependencyTree(Token("given", "VBN", "give"), Seq(
        (DependencyTree(Token("replace", "VB", "replace"), Seq()), "nsubj"),
        (DependencyTree(Token("off", "RP", "off"), Seq()), "prt"),
        (DependencyTree(Token("gas", "NN", "gas"), Seq(
            (DependencyTree(Token("replace", "VB", "replace"), Seq()), "det"))), "dobj")))

    val toReplace = DependencyTree(Token("replace", "VB", "replace"), Seq())
    val replaceWith =
      DependencyTree(Token("replaced", "VBD", "replaced"), Seq(
          (DependencyTree(Token("ha!", "!!", "ha!"), Seq()), "funny")))

    val expectedTree =
      DependencyTree(Token("given", "VBN", "give"), Seq(
        (DependencyTree(Token("replaced", "VBD", "replaced"), Seq(
            (DependencyTree(Token("ha!", "!!", "ha!"), Seq()), "funny"))), "nsubj"),
        (DependencyTree(Token("off", "RP", "off"), Seq()), "prt"),
        (DependencyTree(Token("gas", "NN", "gas"), Seq(
            (DependencyTree(Token("replaced", "VBD", "replaced"), Seq(
                (DependencyTree(Token("ha!", "!!", "ha!"), Seq()), "funny"))), "det"))), "dobj")))

    transformers.replaceTree(tree, toReplace, replaceWith) should be(expectedTree)
  }

  "removeChild" should "leave the rest of the tree intact, and remove one child" in {
    val tree =
      DependencyTree(Token("given", "VBN", "give"), Seq(
        (DependencyTree(Token("plants", "NNS", "plant"), Seq()), "nsubj"),
        (DependencyTree(Token("off", "RP", "off"), Seq()), "prt"),
        (DependencyTree(Token("gas", "NN", "gas"), Seq(
            (DependencyTree(Token("Which", "WDT", "which"), Seq()), "det"))), "dobj")))

    val childLabel = "prt"

    val expectedTree =
      DependencyTree(Token("given", "VBN", "give"), Seq(
        (DependencyTree(Token("plants", "NNS", "plant"), Seq()), "nsubj"),
        (DependencyTree(Token("gas", "NN", "gas"), Seq(
            (DependencyTree(Token("Which", "WDT", "which"), Seq()), "det"))), "dobj")))

    transformers.removeChild(tree, childLabel) should be(expectedTree)
  }

  "removeTree" should "find matching trees, and remove them" in {
    val tree =
      DependencyTree(Token("given", "VBN", "give"), Seq(
        (DependencyTree(Token("remove", "VB", "remove"), Seq()), "nsubj"),
        (DependencyTree(Token("off", "RP", "off"), Seq()), "prt"),
        (DependencyTree(Token("gas", "NN", "gas"), Seq(
            (DependencyTree(Token("remove", "VB", "remove"), Seq()), "det"))), "dobj")))

    val toRemove = DependencyTree(Token("remove", "VB", "remove"), Seq())

    val expectedTree =
      DependencyTree(Token("given", "VBN", "give"), Seq(
        (DependencyTree(Token("off", "RP", "off"), Seq()), "prt"),
        (DependencyTree(Token("gas", "NN", "gas"), Seq()), "dobj")))

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
      DependencyTree(Token("covered", "VBN", "cover"), Seq(
        (DependencyTree(Token("water", "NN", "water"), Seq()), "nsubj"),
        (DependencyTree(Token("Most", "JJS", "most"), Seq(
            (DependencyTree(Token("Earth", "NNP", "Earth"), Seq()), "prep_of"))), "dobj")))
    )
    val tree2 = sentenceTrees("Which gas is given off by plants?")
    transformers.UndoPassivization.transform(tree2) should be(
      DependencyTree(Token("given", "VBN", "give"), Seq(
        (DependencyTree(Token("plants", "NNS", "plant"), Seq()), "nsubj"),
        (DependencyTree(Token("off", "RP", "off"), Seq()), "prt"),
        (DependencyTree(Token("gas", "NN", "gas"), Seq(
            (DependencyTree(Token("Which", "WDT", "which"), Seq()), "det"))), "dobj")))
    )
  }

  it should "leave trees alone when there is no passive" in {
    val tree = sentenceTrees("Which of these is an example of liquid water?")
    transformers.UndoPassivization.transform(tree) should be(tree)
  }

  "ReplaceWhPhrase" should "find the wh-phrase, then replace it with a given tree" in {
    val answerTree = DependencyTree(Token("answer", "NN", "answer"), Seq())
    val tree1 = sentenceTrees("Which gas is given off by plants?")
    new transformers.ReplaceWhPhrase(answerTree).transform(tree1) should be(
      DependencyTree(Token("given", "VBN", "give"), Seq(
        (answerTree, "nsubjpass"),
        (DependencyTree(Token("is", "VBZ", "be"), Seq()), "auxpass"),
        (DependencyTree(Token("off", "RP", "off"), Seq()), "prt"),
        (DependencyTree(Token("plants", "NNS", "plant"), Seq()), "agent")
      ))
    )
    val tree2 = sentenceTrees("Which of these is an example of liquid water?")
    new transformers.ReplaceWhPhrase(answerTree).transform(tree2) should be(
      DependencyTree(Token("is", "VBZ", "be"), Seq(
        (answerTree, "dep"),
        (DependencyTree(Token("example", "NN", "example"), Seq(
          (DependencyTree(Token("an", "DT", "a"), Seq()), "det"),
          (DependencyTree(Token("water", "NN", "water"), Seq(
            (DependencyTree(Token("liquid", "JJ", "liquid"), Seq()), "amod"))), "prep_of"))), "nsubj")))
    )
  }

  it should "do nothing on a tree with no wh-phrase" in {
    val answerTree = DependencyTree(Token("answer", "NN", "answer"), Seq())
    val tree = sentenceTrees("Most of Earth is covered by water.")
    new transformers.ReplaceWhPhrase(answerTree).transform(tree) should be(tree)
  }
}

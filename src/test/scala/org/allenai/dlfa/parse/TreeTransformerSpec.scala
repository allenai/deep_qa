package org.allenai.dlfa.parse

import org.scalatest._

class TreeTransformerSpec extends FlatSpecLike with Matchers {

  // These trees are tricky to write down like this...  Hopefully the structure is clear from this
  // code formatting.
  val sentenceTrees: Map[String, DependencyTree] = Map(
    "Cells contain genetic material called DNA." ->
      DependencyTree(Token("contain", "VBP", "contain", 2), Seq(
        (DependencyTree(Token("Cells", "NNS", "cell", 1), Seq()), "nsubj"),
        (DependencyTree(Token("material", "NN", "material", 4), Seq(
          (DependencyTree(Token("genetic", "JJ", "genetic", 3), Seq()), "amod"),
          (DependencyTree(Token("called", "VBN", "call", 5), Seq(
            (DependencyTree(Token("DNA", "NN", "dna", 6), Seq()), "dobj"))), "vmod"))), "dobj"))),
    "Most of Earth is covered by water." ->
      DependencyTree(Token("covered", "VBN", "cover", 5), Seq(
        (DependencyTree(Token("Most", "JJS", "most", 1), Seq(
          (DependencyTree(Token("Earth", "NNP", "Earth", 3), Seq()), "prep_of"))), "nsubjpass"),
        (DependencyTree(Token("is", "VBZ", "be", 4), Seq()), "auxpass"),
        (DependencyTree(Token("water", "NN", "water", 7), Seq()), "agent"))),
    "Which gas is given off by plants?" ->
      DependencyTree(Token("given", "VBN", "give", 4), Seq(
        (DependencyTree(Token("gas", "NN", "gas", 2), Seq(
          (DependencyTree(Token("Which", "WDT", "which", 1), Seq()), "det"))), "nsubjpass"),
        (DependencyTree(Token("is", "VBZ", "be", 3), Seq()), "auxpass"),
        (DependencyTree(Token("off", "RP", "off", 5), Seq()), "prt"),
        (DependencyTree(Token("plants", "NNS", "plant", 7), Seq()), "agent"))),
    "Which of these is an example of liquid water?" ->
      DependencyTree(Token("is", "VBZ", "be", 4), Seq(
        (DependencyTree(Token("Which", "WDT", "which", 1), Seq(
          (DependencyTree(Token("these", "DT", "these", 3), Seq()), "prep_of"))), "dep"),
        (DependencyTree(Token("example", "NN", "example", 6), Seq(
          (DependencyTree(Token("an", "DT", "a", 5), Seq()), "det"),
          (DependencyTree(Token("water", "NN", "water", 9), Seq(
            (DependencyTree(Token("liquid", "JJ", "liquid", 8), Seq()), "amod"))), "prep_of"))), "nsubj"))),
    "The seeds of an oak come from the fruit." ->
      DependencyTree(Token("come", "VBN", "come", 6), Seq(
        (DependencyTree(Token("seeds", "NNS", "seed", 2), Seq(
          (DependencyTree(Token("The", "DT", "the", 1), Seq()), "det"),
          (DependencyTree(Token("oak", "NN", "oak", 5), Seq(
            (DependencyTree(Token("an", "DT", "a", 4), Seq()), "det"))), "prep_of"))), "nsubj"),
        (DependencyTree(Token("fruit", "NN", "fruit", 9), Seq(
          (DependencyTree(Token("the", "DT", "the", 8), Seq()), "det"))), "prep_from"))),
    "Most erosion at a beach is caused by waves." ->
      DependencyTree(Token("caused", "VBN", "cause", 7), Seq(
        (DependencyTree(Token("erosion", "NN", "erosion", 2), Seq(
          (DependencyTree(Token("Most", "JJS", "most", 1), Seq()), "amod"),
          (DependencyTree(Token("beach", "NN", "beach", 5), Seq(
            (DependencyTree(Token("a", "DT", "a", 4), Seq()), "det"))), "prep_at"))), "nsubjpass"),
        (DependencyTree(Token("is", "VBZ", "be", 6), Seq()), "auxpass"),
        (DependencyTree(Token("waves", "NNS", "wave", 9), Seq()), "agent"))),
    "Water, wind and animals cause erosion." ->
      DependencyTree(Token("cause", "VB", "cause", 6), Seq(
        (DependencyTree(Token("Water", "NN", "water", 1), Seq(
          (DependencyTree(Token("wind", "NN", "wind", 3), Seq()), "conj_and"),
          (DependencyTree(Token("animals", "NNS", "animal", 5), Seq()), "conj_and"))), "nsubj"),
        (DependencyTree(Token("wind", "NN", "wind", 3), Seq()), "nsubj"),
        (DependencyTree(Token("animals", "NNS", "animal", 5), Seq()), "nsubj"),
        (DependencyTree(Token("erosion", "NN", "erosion", 7), Seq()), "dobj"))),
    "This process, photosynthesis, liberates oxygen." ->
      DependencyTree(Token("liberates", "VBZ", "liberate", 6), Seq(
        (DependencyTree(Token("process", "NN", "process", 2), Seq(
          (DependencyTree(Token("This", "DT", "this", 1), Seq()), "det"),
          (DependencyTree(Token("photosynthesis", "NN", "photosynthesis", 4), Seq()), "appos"))), "nsubj"),
        (DependencyTree(Token("oxygen", "NN", "oxygen", 7), Seq()), "dobj"))),
    "What is the male part of a flower called?" ->
      DependencyTree(Token("called", "VBN", "call", 9), Seq(
        (DependencyTree(Token("What", "WP", "what", 1), Seq()), "dobj"),
        (DependencyTree(Token("is", "VBZ", "be", 2), Seq()), "auxpass"),
        (DependencyTree(Token("part", "NN", "part", 5), Seq(
          (DependencyTree(Token("the", "DT", "the", 3), Seq()), "det"),
          (DependencyTree(Token("male", "JJ", "male", 4), Seq()), "amod"),
          (DependencyTree(Token("flower", "NN", "flower", 8), Seq(
            (DependencyTree(Token("a", "DT", "a", 7), Seq()), "det"))), "prep_of"))), "nsubjpass")))

  )

  "getParent" should "return the correct parent tree" in {
    val tree = sentenceTrees("This process, photosynthesis, liberates oxygen.")
    val childTree =
      DependencyTree(Token("photosynthesis", "NN", "photosynthesis", 4), Seq())
    val parentTree =
      DependencyTree(Token("process", "NN", "process", 2), Seq(
        (DependencyTree(Token("This", "DT", "this", 1), Seq()), "det"),
        (DependencyTree(Token("photosynthesis", "NN", "photosynthesis", 4), Seq()), "appos")))
    transformers.getParent(tree, childTree) should be(Some(parentTree))
  }

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

  "swapChildrenOrder" should "fix token indices and swap children order in a simple example" in {
    val tree =
      DependencyTree(Token("called", "VBN", "call", 5), Seq(
        (DependencyTree(Token("What", "WP", "what", 1), Seq()), "dobj"),
        (DependencyTree(Token("is", "VBZ", "be", 2), Seq()), "auxpass"),
        (DependencyTree(Token("part", "NN", "part", 4), Seq(
          (DependencyTree(Token("the", "DT", "the", 3), Seq()), "det"))), "nsubjpass")))

    val expectedTree =
      DependencyTree(Token("called", "VBN", "call", 5), Seq(
        (DependencyTree(Token("What", "WP", "what", 1), Seq()), "dobj"),
        (DependencyTree(Token("part", "NN", "part", 3), Seq(
          (DependencyTree(Token("the", "DT", "the", 2), Seq()), "det"))), "nsubjpass"),
        (DependencyTree(Token("is", "VBZ", "be", 4), Seq()), "auxpass")))

    transformers.swapChildrenOrder(tree, "nsubjpass", "auxpass") should be(expectedTree)
  }

  it should "work on a more complicated example, too" in {
    val tree = sentenceTrees("What is the male part of a flower called?")
    val expectedTree =
      DependencyTree(Token("called", "VBN", "call", 9), Seq(
        (DependencyTree(Token("part", "NN", "part", 3), Seq(
          (DependencyTree(Token("the", "DT", "the", 1), Seq()), "det"),
          (DependencyTree(Token("male", "JJ", "male", 2), Seq()), "amod"),
          (DependencyTree(Token("flower", "NN", "flower", 6), Seq(
            (DependencyTree(Token("a", "DT", "a", 5), Seq()), "det"))), "prep_of"))), "nsubjpass"),
        (DependencyTree(Token("is", "VBZ", "be", 7), Seq()), "auxpass"),
        (DependencyTree(Token("What", "WP", "what", 8), Seq()), "dobj")
      ))
    transformers.swapChildrenOrder(tree, "nsubjpass", "dobj") should be(expectedTree)
  }

  it should "update the root's token index if it is between the moved children" in {
    val tree =
      DependencyTree(Token("is", "VBZ", "be", 2), Seq(
        (DependencyTree(Token("What", "WP", "what", 1), Seq()), "dobj"),
        (DependencyTree(Token("stage", "NN", "stage", 4), Seq(
          (DependencyTree(Token("the", "DT", "the", 3), Seq()), "det"))), "nsubj")))
    val expectedTree =
      DependencyTree(Token("is", "VBZ", "be", 3), Seq(
        (DependencyTree(Token("stage", "NN", "stage", 2), Seq(
          (DependencyTree(Token("the", "DT", "the", 1), Seq()), "det"))), "nsubj"),
        (DependencyTree(Token("What", "WP", "what", 4), Seq()), "dobj")))
    transformers.swapChildrenOrder(tree, "nsubj", "dobj") should be(expectedTree)
  }

  it should "do nothing if the children aren't found" in {
    val tree = sentenceTrees("What is the male part of a flower called?")
    transformers.swapChildrenOrder(tree, "fake", "dobj") should be(tree)
    transformers.swapChildrenOrder(tree, "dobj", "fake") should be(tree)
  }

  it should "do nothing if the children are in the correct order" in {
    val tree = sentenceTrees("What is the male part of a flower called?")
    transformers.swapChildrenOrder(tree, "dobj", "auxpass") should be(tree)
  }

  "moveHeadToLeftOfChild" should "correctly update token indices in the tree" in {
    val tree = sentenceTrees("What is the male part of a flower called?")
    val expectedTree =
      DependencyTree(Token("called", "VBN", "call", 1), Seq(
        (DependencyTree(Token("What", "WP", "what", 2), Seq()), "dobj"),
        (DependencyTree(Token("is", "VBZ", "be", 3), Seq()), "auxpass"),
        (DependencyTree(Token("part", "NN", "part", 6), Seq(
          (DependencyTree(Token("the", "DT", "the", 4), Seq()), "det"),
          (DependencyTree(Token("male", "JJ", "male", 5), Seq()), "amod"),
          (DependencyTree(Token("flower", "NN", "flower", 9), Seq(
            (DependencyTree(Token("a", "DT", "a", 8), Seq()), "det"))), "prep_of"))), "nsubjpass")
      ))
    transformers.moveHeadToLeftOfChild(tree, "dobj") should be(expectedTree)
  }

  it should "do nothing if the head is already on the left" in {
    val tree = sentenceTrees("Which gas is given off by plants?")
    transformers.moveHeadToLeftOfChild(tree, "agent") should be(tree)
  }

  it should "do nothing if the child isn't found" in {
    val tree = sentenceTrees("Which gas is given off by plants?")
    transformers.moveHeadToLeftOfChild(tree, "fake") should be(tree)
  }

  "findWhPhrase" should "find the correct wh-phrase" in {
    val tree1 = sentenceTrees("Which gas is given off by plants?")
    transformers.findWhPhrase(tree1) should be(Some(tree1.children(0)._1))
    val tree2 = sentenceTrees("Which of these is an example of liquid water?")
    transformers.findWhPhrase(tree2) should be(Some(tree2.children(0)._1))
    val tree3 = sentenceTrees("What is the male part of a flower called?")
    transformers.findWhPhrase(tree3) should be(Some(tree3.children(0)._1))
  }

  "RemoveConjunctionDuplicates" should "remove duplicates in a tree with conjunctions" in {
    val tree = sentenceTrees("Water, wind and animals cause erosion.")
    val expectedTree =
      DependencyTree(Token("cause", "VB", "cause", 6), Seq(
        (DependencyTree(Token("Water", "NN", "water", 1), Seq(
          (DependencyTree(Token("wind", "NN", "wind", 3), Seq()), "conj_and"),
          (DependencyTree(Token("animals", "NNS", "animal", 5), Seq()), "conj_and"))), "nsubj"),
        (DependencyTree(Token("erosion", "NN", "erosion", 7), Seq()), "dobj")))
    transformers.RemoveConjunctionDuplicates.transform(tree) should be(expectedTree)
  }

  "UndoWhMovement" should "undo wh-movement" in {
    val tree = sentenceTrees("What is the male part of a flower called?")
    val expecedTree =
      DependencyTree(Token("called", "VBN", "call", 8), Seq(
        (DependencyTree(Token("part", "NN", "part", 3), Seq(
          (DependencyTree(Token("the", "DT", "the", 1), Seq()), "det"),
          (DependencyTree(Token("male", "JJ", "male", 2), Seq()), "amod"),
          (DependencyTree(Token("flower", "NN", "flower", 6), Seq(
            (DependencyTree(Token("a", "DT", "a", 5), Seq()), "det"))), "prep_of"))), "nsubjpass"),
        (DependencyTree(Token("is", "VBZ", "be", 7), Seq()), "auxpass"),
        (DependencyTree(Token("What", "WP", "what", 9), Seq()), "dobj")
      ))
    transformers.UndoWhMovement.transform(tree) should be(expecedTree)
  }

  it should "not get into an infinite loop" in {
    val tree =
      DependencyTree(Token("happen", "VB", "happen", 16), Seq(
        (DependencyTree(Token("cleared", "VBN", "clear", 6), Seq(
          (DependencyTree(Token("If", "IN", "if", 1), Seq()), "mark"),
          (DependencyTree(Token("area", "NN", "area", 4), Seq(
            (DependencyTree(Token("a", "DT", "a", 2), Seq()), "det"),
            (DependencyTree(Token("wooded", "JJ", "wooded", 3), Seq()), "amod"))), "nsubjpass"),
          (DependencyTree(Token("is", "VBZ", "be", 5), Seq()), "auxpass"),
          (DependencyTree(Token("planted", "VBN", "plant", 10), Seq(
            (DependencyTree(Token("corn", "NN", "corn", 8), Seq()), "nsubjpass"),
            (DependencyTree(Token("is", "VBZ", "be", 9), Seq()), "auxpass"))), "conj_and"))), "advcl"),
        (DependencyTree(Token("planted", "VBN", "plant", 10), Seq(
          (DependencyTree(Token("corn", "NN", "corn", 8), Seq()), "nsubjpass"),
          (DependencyTree(Token("is", "VBZ", "be", 9), Seq()), "auxpass"))), "advcl"),
        (DependencyTree(Token("what", "WP", "what", 12), Seq()), "dobj"),
        (DependencyTree(Token("will", "MD", "will", 13), Seq()), "aux"),
        (DependencyTree(Token("MOST", "JJS", "most", 14), Seq(
          (DependencyTree(Token("likely", "JJ", "likely", 15), Seq()), "amod"))), "nsubj")))
    // Note that the token indices here are off by one in a few places, but the ordering is still
    // correct.  That is because it's really difficult to handle clause-separating commas
    // correctly.  They aren't in the dependency tree that Stanford gives us, and I don't want to
    // bother figuring out where they go.  So, we'll just deal with this numbering oddity, as the
    // code that uses this should still work if the ordering is correct.
    val expectedTree =
      DependencyTree(Token("happen", "VB", "happen", 5), Seq(
        (DependencyTree(Token("MOST", "JJS", "most", 1), Seq(
          (DependencyTree(Token("likely", "JJ", "likely", 2), Seq()), "amod"))), "nsubj"),
        (DependencyTree(Token("will", "MD", "will", 4), Seq()), "aux"),
        (DependencyTree(Token("what", "WP", "what", 6), Seq()), "dobj"),
        (DependencyTree(Token("cleared", "VBN", "clear", 12), Seq(
          (DependencyTree(Token("If", "IN", "if", 7), Seq()), "mark"),
          (DependencyTree(Token("area", "NN", "area", 10), Seq(
            (DependencyTree(Token("a", "DT", "a", 8), Seq()), "det"),
            (DependencyTree(Token("wooded", "JJ", "wooded", 9), Seq()), "amod"))), "nsubjpass"),
          (DependencyTree(Token("is", "VBZ", "be", 11), Seq()), "auxpass"),
          (DependencyTree(Token("planted", "VBN", "plant", 16), Seq(
            (DependencyTree(Token("corn", "NN", "corn", 14), Seq()), "nsubjpass"),
            (DependencyTree(Token("is", "VBZ", "be", 15), Seq()), "auxpass"))), "conj_and"))), "advcl")))
    transformers.UndoWhMovement.transform(tree) should be(expectedTree)
  }

  "MakeCopulaHead" should "rotate the tree so the copula is the head" in {
    val tree =
      DependencyTree(Token("example", "NN", "example", 4), Seq(
        (DependencyTree(Token("Photosynthesis", "NN", "photosynthesis", 1), Seq()), "nsubj"),
        (DependencyTree(Token("is", "VBZ", "be", 2), Seq()), "cop"),
        (DependencyTree(Token("an", "DT", "a", 3), Seq()), "det"),
        (DependencyTree(Token("process", "NN", "process", 7), Seq(
          (DependencyTree(Token("a", "DT", "a", 6), Seq()), "det"))), "prep_of")))
    val expectedTree =
      DependencyTree(Token("is", "VBZ", "be", 2), Seq(
        (DependencyTree(Token("Photosynthesis", "NN", "photosynthesis", 1), Seq()), "nsubj"),
        (DependencyTree(Token("example", "NN", "example", 4), Seq(
          (DependencyTree(Token("an", "DT", "a", 3), Seq()), "det"),
          (DependencyTree(Token("process", "NN", "process", 7), Seq(
            (DependencyTree(Token("a", "DT", "a", 6), Seq()), "det"))), "prep_of"))), "dobj")))
    transformers.MakeCopulaHead.transform(tree) should be(expectedTree)
  }

  it should "keep adverbs attached to the root" in {
    val tree =
      DependencyTree(Token("important", "JJ", "important", 4), Seq(
        (DependencyTree(Token("Why", "WRB", "why", 1), Seq()), "advmod"),
        (DependencyTree(Token("is", "VBZ", "be", 2), Seq()), "cop"),
        (DependencyTree(Token("competition", "NN", "competition", 3), Seq()), "nsubj")))
    val expectedTree =
      DependencyTree(Token("is", "VBZ", "be", 2), Seq(
        (DependencyTree(Token("Why", "WRB", "why", 1), Seq()), "advmod"),
        (DependencyTree(Token("competition", "NN", "competition", 3), Seq()), "nsubj"),
        (DependencyTree(Token("important", "JJ", "important", 4), Seq()), "dobj")))
    transformers.MakeCopulaHead.transform(tree) should be(expectedTree)
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

  "RemoveSuperlatives" should "remove \"most\" when it's used as an adjective" in {
    val tree = sentenceTrees("Most erosion at a beach is caused by waves.")
    transformers.RemoveSuperlatives.transform(tree) should be(
      DependencyTree(Token("caused", "VBN", "cause", 7), Seq(
        (
          DependencyTree(Token("erosion", "NN", "erosion", 2), Seq(
            (DependencyTree(Token("beach", "NN", "beach", 5), Seq(
              (DependencyTree(Token("a", "DT", "a", 4), Seq()), "det"))), "prep_at")
          )),
          "nsubjpass"
        ),
        (DependencyTree(Token("is", "VBZ", "be", 6), Seq()), "auxpass"),
        (DependencyTree(Token("waves", "NNS", "wave", 9), Seq()), "agent")
      ))
    )
  }

  it should "keep \"most\" when it's not used as an adjective" in {
    val tree = sentenceTrees("Most of Earth is covered by water.")
    transformers.RemoveSuperlatives.transform(tree) should be(tree)
  }

  "SplitConjunctions" should "return two trees when the subject has a conjunction" in {
    val tree = sentenceTrees("Water, wind and animals cause erosion.")
    transformers.SplitConjunctions.transform(tree) should be(Set(
      DependencyTree(Token("cause", "VB", "cause", 6), Seq(
        (DependencyTree(Token("Water", "NN", "water", 1), Seq()), "nsubj"),
        (DependencyTree(Token("erosion", "NN", "erosion", 7), Seq()), "dobj")
      )),
      DependencyTree(Token("cause", "VB", "cause", 6), Seq(
        (DependencyTree(Token("wind", "NN", "wind", 3), Seq()), "nsubj"),
        (DependencyTree(Token("erosion", "NN", "erosion", 7), Seq()), "dobj")
      )),
      DependencyTree(Token("cause", "VB", "cause", 6), Seq(
        (DependencyTree(Token("animals", "NNS", "animal", 5), Seq()), "nsubj"),
        (DependencyTree(Token("erosion", "NN", "erosion", 7), Seq()), "dobj")
      ))
    ))
  }

  it should "work with multiple conjunctions" in {
    val tree =
      DependencyTree(Token("cause", "VB", "cause", 6), Seq(
        (DependencyTree(Token("Water", "NN", "water", 1), Seq(
          (DependencyTree(Token("wind", "NN", "wind", 3), Seq()), "conj_and"))), "nsubj"),
        (DependencyTree(Token("wind", "NN", "wind", 3), Seq()), "nsubj"),
        (DependencyTree(Token("erosion", "NN", "erosion", 7), Seq(
          (DependencyTree(Token("decay", "NN", "decay", 9), Seq()), "conj_and"))), "dobj"),
        (DependencyTree(Token("decay", "NN", "decay", 9), Seq()), "dobj")))
    transformers.SplitConjunctions.transform(tree) should be(Set(
      DependencyTree(Token("cause", "VB", "cause", 6), Seq(
        (DependencyTree(Token("Water", "NN", "water", 1), Seq()), "nsubj"),
        (DependencyTree(Token("erosion", "NN", "erosion", 7), Seq()), "dobj"))),
      DependencyTree(Token("cause", "VB", "cause", 6), Seq(
        (DependencyTree(Token("wind", "NN", "wind", 3), Seq()), "nsubj"),
        (DependencyTree(Token("erosion", "NN", "erosion", 7), Seq()), "dobj"))),
      DependencyTree(Token("cause", "VB", "cause", 6), Seq(
        (DependencyTree(Token("Water", "NN", "water", 1), Seq()), "nsubj"),
        (DependencyTree(Token("decay", "NN", "decay", 9), Seq()), "dobj"))),
      DependencyTree(Token("cause", "VB", "cause", 6), Seq(
        (DependencyTree(Token("wind", "NN", "wind", 3), Seq()), "nsubj"),
        (DependencyTree(Token("decay", "NN", "decay", 9), Seq()), "dobj")))
    ))
  }

  "SplitAppositives" should "return three trees when there is an appositive" in {
    val tree = sentenceTrees("This process, photosynthesis, liberates oxygen.")
    transformers.SplitAppositives.transform(tree) should be(Set(
      DependencyTree(Token("liberates", "VBZ", "liberate", 6), Seq(
        (DependencyTree(Token("photosynthesis", "NN", "photosynthesis", 4), Seq()), "nsubj"),
        (DependencyTree(Token("oxygen", "NN", "oxygen", 7), Seq()), "dobj"))),
      DependencyTree(Token("liberates", "VBZ", "liberate", 6), Seq(
        (DependencyTree(Token("process", "NN", "process", 2), Seq(
          (DependencyTree(Token("This", "DT", "this", 1), Seq()), "det"))), "nsubj"),
        (DependencyTree(Token("oxygen", "NN", "oxygen", 7), Seq()), "dobj"))),
      DependencyTree(Token("appos", "VB", "appos", 0), Seq(
        (DependencyTree(Token("process", "NN", "process", 2), Seq(
          (DependencyTree(Token("This", "DT", "this", 1), Seq()), "det"))), "nsubj"),
        (DependencyTree(Token("photosynthesis", "NN", "photosynthesis", 4), Seq()), "dobj")))
    ))
  }
}

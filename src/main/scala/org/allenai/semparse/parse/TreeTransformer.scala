package org.allenai.semparse.parse

trait TreeTransformer {
  def transform(tree: DependencyTree): DependencyTree
}

abstract class BaseTransformer extends TreeTransformer {
  def transformChildren(tree: DependencyTree): DependencyTree = {
    DependencyTree(tree.token, tree.children.map(childWithLabel => {
      val label = childWithLabel._2
      val child = childWithLabel._1
      (transform(child), label)
    }))
  }
}

object transformers {

  /**
   * Replaces the _first_ occurrence of a child with label childLabel with newChild.  This is NOT
   * recursive, and will throw an error if the given child label is not found.
   *
   * NOTE: calling a yield method after transforming a tree with this is not safe!  It will not
   * work as you expect, as indices aren't currently updated in this method.
   */
  def replaceChild(
    tree: DependencyTree,
    childLabel: String,
    newChild: DependencyTree,
    newLabel: String
  ): DependencyTree = {
    val childWithIndex = tree.children.zipWithIndex.find(_._1._2 == childLabel)
    childWithIndex match {
      case None => throw new IllegalStateException(s"Didn't find child with label $childLabel")
      case Some((child, indexToReplace)) => {
        val newChildren = tree.children.zipWithIndex.map(c => {
          val index = c._2
          if (index == indexToReplace) {
            (newChild, newLabel)
          } else {
            c._1
          }
        })
        DependencyTree(tree.token, newChildren)
      }
    }
  }

  /**
   * Traverses the tree to find matching subtrees, and replaces them with the given tree.
   *
   * NOTE: calling a yield method after transforming a tree with this is not safe!  It will not
   * work as you expect, as indices aren't currently updated in this method.
   */
  def replaceTree(
    tree: DependencyTree,
    treeToReplace: DependencyTree,
    replaceWith: DependencyTree
  ): DependencyTree = {
    if (tree == treeToReplace) {
      replaceWith
    } else {
      DependencyTree(tree.token, tree.children.map(childWithLabel => {
        val label = childWithLabel._2
        val child = childWithLabel._1
        (replaceTree(child, treeToReplace, replaceWith), label)
      }))
    }
  }

  /**
   * Removes the _first_ occurrence of a child with label childLabel.  This is NOT recursive, and
   * will throw an error if the given child label is not found.
   */
  def removeChild(tree: DependencyTree, childLabel: String): DependencyTree = {
    val childWithIndex = tree.children.zipWithIndex.find(_._1._2 == childLabel)
    childWithIndex match {
      case None => throw new IllegalStateException(s"Didn't find child with label $childLabel")
      case Some((child, indexToRemove)) => {
        val newChildren = tree.children.zipWithIndex.flatMap(c => {
          val index = c._2
          if (index == indexToRemove) {
            Seq()
          } else {
            Seq(c._1)
          }
        })
        DependencyTree(tree.token, newChildren)
      }
    }
  }

  /**
   * Assuming subtree is part of tree, find the parent of subtree in tree.  Returns None if the
   * subtree is not present or if it is the root.
   */
  def getParent(tree: DependencyTree, subtree: DependencyTree): Option[DependencyTree] = {
    tree.children.find(_._1 == subtree) match {
      case Some(child) => Some(tree)
      case None => {
        tree.children.map(c => getParent(c._1, subtree)).flatten.headOption
      }
    }
  }

  /**
   * Traverses the tree to find matching subtrees, and removes them.
   */
  def removeTree(tree: DependencyTree, treeToRemove: DependencyTree): DependencyTree = {
    DependencyTree(tree.token, tree.children.flatMap(childWithLabel => {
      val label = childWithLabel._2
      val child = childWithLabel._1
      if (child == treeToRemove) {
        Seq()
      } else {
        Seq((removeTree(child, treeToRemove), label))
      }
    }))
  }

  /**
   * Adds a child node with the given label to the end of the tree's children list.
   */
  def addChild(tree: DependencyTree, child: DependencyTree, label: String): DependencyTree = {
    DependencyTree(tree.token, tree.children ++ Seq((child, label)))
  }

  /**
   *  This method looks for children with the given labels, and checks to see if they are already
   *  in the given order.  If both children exist, and they are in the wrong order, it swaps them
   *  and updates their token indices (and the indices of all intermediate children).
   *
   *  Note that this method assumes the children are sorted according to token indices.  This is
   *  true of a freshly-constructed tree, but it's not necessarily true of a tree that's been
   *  through some transformations (e.g., the addChild(), replaceChild(), and replaceTree()
   *  methods).
   */
  def swapChildrenOrder(
    tree: DependencyTree,
    firstChildLabel: String,
    secondChildLabel: String
  ): DependencyTree = {
    val firstChild = tree.getChildWithLabel(firstChildLabel) match {
      case None => return tree
      case Some(c) => (c, firstChildLabel)
    }
    val secondChild = tree.getChildWithLabel(secondChildLabel) match {
      case None => return tree
      case Some(c) => (c, secondChildLabel)
    }

    val firstChildIndex = tree.children.indexOf(firstChild)
    val secondChildIndex = tree.children.indexOf(secondChild)
    if (firstChildIndex < secondChildIndex) return tree

    // Ok, we found the right children, and they are out of order.  Now we swap them.
    val childrenBefore = tree.children.take(secondChildIndex)
    val childrenAfter = tree.children.drop(firstChildIndex + 1)
    val childrenBetween = tree.children.drop(secondChildIndex + 1).take(firstChildIndex - secondChildIndex - 1)

    // And now we fix the indices.
    val firstChildStartToken = firstChild._1.tokenStartIndex
    val secondChildStartToken = secondChild._1.tokenStartIndex

    val shiftDown = secondChildStartToken - firstChildStartToken
    val newFirstChild = (firstChild._1.shiftIndicesBy(shiftDown), firstChild._2)

    val shiftUp = firstChildStartToken - secondChildStartToken + firstChild._1.numTokens - 1
    val newSecondChild = (secondChild._1.shiftIndicesBy(shiftUp), secondChild._2)

    val shiftBetween = firstChild._1.numTokens - secondChild._1.numTokens
    val newChildrenBetween = childrenBetween.map(c => (c._1.shiftIndicesBy(shiftBetween), c._2))

    val newChildren = childrenBefore ++ Seq(newFirstChild) ++ newChildrenBetween ++ Seq(newSecondChild) ++ childrenAfter

    // Finally, check if we need to update the root token index.
    val rootIndex = tree.token.index
    val newToken = if (rootIndex > secondChildStartToken && rootIndex < firstChildStartToken) {
      tree.token.shiftIndexBy(shiftBetween)
    } else {
      tree.token
    }

    DependencyTree(newToken, newChildren)
  }

  /**
   * This method looks for a child with a given label, and checks to see if it is to the left of
   * the head token (i.e., the child index is lower than the head index).  If it is, we update the
   * indices of the head token and all other tokens in the tree between the head and the child, so
   * that the head is immediately to the left of the given child.
   *
   *  Note that this method assumes the children are sorted according to token indices.  This is
   *  true of a freshly-constructed tree, but it's not necessarily true of a tree that's been
   *  through some transformations (e.g., the addChild(), replaceChild(), and replaceTree()
   *  methods).
   */
  def moveHeadToLeftOfChild(tree: DependencyTree, childLabel: String): DependencyTree = {
    val child = tree.getChildWithLabel(childLabel) match {
      case None => return tree
      case Some(c) => (c, childLabel)
    }
    val childStartToken = child._1.tokenStartIndex
    val headToken = tree.token.index

    if (headToken < childStartToken) return tree

    // Ok, we found the child, and it's on the wrong side of the head.  We need to swap the indices
    // of the head and the children inbetween the head and this child.
    val childrenBefore = tree.children.filter(_._1.tokenStartIndex < childStartToken)
    val childrenAfter = tree.children.filter(_._1.tokenStartIndex > headToken)
    val childrenBetween = tree.children.filter(c => c._1.tokenStartIndex >= childStartToken && c._1.tokenStartIndex < headToken)

    val newToken = tree.token.withNewIndex(childStartToken)
    val newChildrenBetween = childrenBetween.map(c => (c._1.shiftIndicesBy(1), c._2))
    val newChildren = childrenBefore ++ newChildrenBetween ++ childrenAfter
    DependencyTree(newToken, newChildren)
  }

  /**
   * Finds the subtree that corresponds to a wh-phrase, which would be replaced if we were turning
   * a question into a declarative sentence.  We currently assume that all wh-phrases are of the
   * form "(Which|What) NP? VP", where "which" or "what" is the (determiner) child of the NP, or
   * "Which of NP VP", where "which" is the (still determiner) head of the NP.
   */
  def findWhPhrase(tree: DependencyTree): Option[DependencyTree] = {
    if (tree.isWhPhrase) {
      // We'll check this as a base case; we'll have already caught cases like "which gas", where
      // "gas" is the head, higher up in the tree.
      Some(tree)
    } else {
      val hasWhChild = tree.children.exists(c => c._2 == "det" && c._1.isWhPhrase && c._1.children.size == 0)
      if (hasWhChild) {
        Some(tree)
      } else {
        val childrenWhPhrases = tree.children.map(c => findWhPhrase(c._1))
        val successes = childrenWhPhrases.flatMap(_.toList)
        if (successes.size == 0) {
          None
        } else if (successes.size == 1) {
          Some(successes.head)
        } else {
          throw new IllegalStateException("found multiple wh-phrases - is this a real sentence?")
        }
      }
    }
  }

  object UndoPassivization extends BaseTransformer {
    override def transform(tree: DependencyTree): DependencyTree = {
      val children = tree.children
      if (children.size < 3) {
        transformChildren(tree)
      } else {
        val labels = children.map(_._2).toSet
        if (labels.contains("nsubjpass") && labels.contains("agent") && labels.contains("auxpass")) {
          val nsubjpass = children.find(_._2 == "nsubjpass").get._1
          val agent = children.find(_._2 == "agent").get._1
          // We take what was the agent, and make it the subject (putting the agent tree where
          // "nsubjpass" was).
          var transformed = replaceChild(tree, "nsubjpass", agent, "nsubj")
          // We take what was the object and make it the subject (putting the nsubjpass tree where
          // "agent" was).
          transformed = replaceChild(transformed, "agent", nsubjpass, "dobj")
          // And we remove the auxpass auxiliary (typically "is" or "was").
          transformed = removeChild(transformed, "auxpass")
          transformChildren(transformed)
        } else {
          transformChildren(tree)
        }
      }
    }
  }

  object MakeCopulaHead extends BaseTransformer {
    override def transform(tree: DependencyTree): DependencyTree = {
      tree.getChildWithLabel("cop") match {
        case None => tree
        case Some(cop) => {
          tree.getChildWithLabel("nsubj") match {
            case None => tree
            case Some(nsubj) => {
              val copulaRemoved = removeChild(tree, "cop")
              val dobj = removeChild(removeChild(tree, "cop"), "nsubj")
              val (addFirst, addSecond) = if (dobj.tokenStartIndex < nsubj.tokenStartIndex) {
                ((dobj, "dobj"), (nsubj, "nsubj"))
              } else {
                ((nsubj, "nsubj"), (dobj, "dobj"))
              }
              addChild(addChild(cop, addFirst._1, addFirst._2), addSecond._1, addSecond._2)
            }
          }
        }
      }
    }
  }

  object RemoveDeterminers extends BaseTransformer {
    override def transform(tree: DependencyTree): DependencyTree = {
      DependencyTree(tree.token, tree.children.flatMap(childWithLabel => {
        val label = childWithLabel._2
        val child = childWithLabel._1
        if (label == "det" && child.isDeterminer && child.children.size == 0) {
          Seq()
        } else {
          Seq((transform(child), label))
        }
      }))
    }
  }

  object CombineParticles extends BaseTransformer {
    override def transform(tree: DependencyTree): DependencyTree = {
      tree.getChildWithLabel("prt") match {
        case None => transformChildren(tree)
        case Some(child) => {
          val newToken = tree.token.combineWith(child.token)
          val newTree = DependencyTree(newToken, tree.children.filterNot(_._2 == "prt"))
          transformChildren(newTree)
        }
      }
    }
  }

  /**
   * Removes a few words that are particular to our science questions.  A question will often say
   * something like, "which of the following is the best conductor of electricity?", with answer
   * options "iron rod", "plastic spoon", and so on.  What we really want to score is "An iron rod
   * is a conductor of electricity" vs "A plastic spoon is a conductor of electricity" - the
   * superlative "best" is captured implicitly in our ranking, and so we don't need it to be part
   * of the logical form that we score.  So we're going to remove a few specific words that capture
   * this notion.
   */
  object RemoveSuperlatives extends BaseTransformer {
    def isMatchingSuperlative(tree: DependencyTree): Boolean = {
      tree.token.lemma == "most" && tree.children.size == 0
    }

    override def transform(tree: DependencyTree): DependencyTree = {
      tree.children.find(c => isMatchingSuperlative(c._1) && c._2 == "amod") match {
        case None => transformChildren(tree)
        case Some((child, label)) => {
          transformChildren(removeTree(tree, child))
        }
      }
    }
  }

  // TODO(matt): I might want to have this return two trees in some cases, for things like "which
  // characteristic is ..." - include an "appos" or "be" tree, along with the replaced tree.
  class ReplaceWhPhrase(replaceWith: DependencyTree) extends BaseTransformer {
    override def transform(tree: DependencyTree) = {
      findWhPhrase(tree) match {
        case None => tree
        case Some(whTree) => replaceTree(tree, whTree, replaceWith)
      }
    }
  }

  object UndoWhMovement extends BaseTransformer {
    val order = Seq("nsubj", "nsubjpass", "aux", "auxpass", "head", "iobj", "dobj", "advmod")
    def childSortKey(child: (DependencyTree, String)) = {
      order.indexOf(child._2) match {
        case -1 => order.size
        case i => i
      }
    }

    def childrenAreSorted(tree: DependencyTree): Boolean = {
      val sorted = tree.children.sortBy(childSortKey)
      tree.children == sorted
    }

    def findLabelsToSwap(tree: DependencyTree): Option[(String, String)] = {
      for (i <- 0 until tree.children.size) {
        val iSortOrder = childSortKey(tree.children(i))
        for (j <- (i + 1) until tree.children.size) {
          val jSortOrder = childSortKey(tree.children(j))
          if (iSortOrder > jSortOrder) return Some((tree.children(j)._2, tree.children(i)._2))
        }
      }
      return None
    }

    def transform(tree: DependencyTree): DependencyTree = {
      findWhPhrase(tree) match {
        case None => tree
        case Some(whTree) => {
          var currentTree = tree
          while (!childrenAreSorted(currentTree)) {
            val (firstLabel, secondLabel) = findLabelsToSwap(currentTree) match {
              case None => throw new RuntimeException("not sure how this happened...")
              case Some((f, s)) => (f, s)
            }
            currentTree = swapChildrenOrder(currentTree, firstLabel, secondLabel)
          }
          for (i <- ((order.indexOf("head") + 1) until order.size).reverse) {
            currentTree = moveHeadToLeftOfChild(currentTree, order(i))
          }
          currentTree
        }
      }
    }
  }

  object SplitConjunctions {
    def findConjunctions(tree: DependencyTree): Set[(DependencyTree, DependencyTree)] = {
      val children = tree.children.toSet
      children.flatMap(childWithLabel => {
        val child = childWithLabel._1
        val label = childWithLabel._2
        if (label == "conj_and") {
          Set((tree, child)) ++ findConjunctions(child)
        } else {
          findConjunctions(child)
        }
      })
    }

    def transform(tree: DependencyTree): Set[DependencyTree] = {
      // Basic outline here:
      // 1. find all conjunctions, paired with the parent node
      // 2. group by the parent node, in case there is more than one conjunction in the sentence
      // 3. for the first conjunction:
      //   4. remove all conjunction trees, forming a tree with just one of the conjoined nodes
      //   5. for each conjunction tree, remove the parent, and replace it with the conjunction tree
      //   6. for each of these constructed trees, recurse

      // Step 1
      val conjunctionTrees = findConjunctions(tree)

      // Step 2
      conjunctionTrees.groupBy(_._1).headOption match {
        case None => Set(tree)
        // Step 3
        case Some((parent, childrenWithParent)) => {
          val children = childrenWithParent.map(_._2)

          // Step 4
          var justFirstConjunction = tree
          var justParent = parent
          for (child <- children) {
            justFirstConjunction = removeTree(justFirstConjunction, child)
            justParent = removeTree(justParent, child)
          }

          // Step 5
          val otherConjunctions = children.map(child => {
            replaceTree(justFirstConjunction, justParent, child)
          })

          // Step 6
          val separatedTrees = Set(justFirstConjunction) ++ otherConjunctions
          separatedTrees.flatMap(transform)
        }
      }
    }
  }

  // This is very similar to SplitConjunctions, but the tree created by the Stanford parser is
  // slightly different, so there are a few minor changes here.
  // TODO(matt): actually, I only needed to change the findConjunctions method, and everything else
  // was the same.  I should make these two transformers share code.
  // Actually, I added a new tree also for each appositive, just containing the appositive
  // relationship, so that it would get extracted correctly in the logical form.
  object SplitAppositives {
    def findAppositives(tree: DependencyTree): Set[(DependencyTree, DependencyTree)] = {
      val children = tree.children.toSet
      children.flatMap(childWithLabel => {
        val child = childWithLabel._1
        val label = childWithLabel._2
        if (label == "appos") {
          Set((tree, child)) ++ findAppositives(child)
        } else {
          findAppositives(child)
        }
      })
    }

    def transform(tree: DependencyTree): Set[DependencyTree] = {
      // Basic outline here:
      // 1. find all appositives, paired with the parent node
      // 2. group by the parent node, in case there is more than one appositives in the sentence
      // 3. for the first appositives:
      //   4. remove all appositives trees, forming a tree with just the head NP
      //   5. for each appositive tree, remove the parent, and replace it with the conjunction tree
      //   6. for each of these constructed trees, recurse
      // 7. add a tree containing the appositive relationship

      // Step 1
      val apposTrees = findAppositives(tree)

      // Step 2
      apposTrees.groupBy(_._1).headOption match {
        case None => Set(tree)
        // Step 3
        case Some((parent, childrenWithParent)) => {
          val children = childrenWithParent.map(_._2)

          // Step 4
          var justFirstAppositive = tree
          var justParent = parent
          for (child <- children) {
            justFirstAppositive = removeTree(justFirstAppositive, child)
            justParent = removeTree(justParent, child)
          }

          // Step 5
          val otherAppositives = children.map(child => {
            replaceTree(justFirstAppositive, justParent, child)
          })

          // Step 6
          val separatedTrees = Set(justFirstAppositive) ++ otherAppositives

          // Step 7
          val appositiveTrees = children.map(child => {
            DependencyTree(Token("appos", "VB", "appos", 0), Seq(
              (justParent, "nsubj"), (child, "dobj")))
          })
          separatedTrees.flatMap(transform) ++ appositiveTrees
        }
      }
    }
  }

  object RemoveBareCCs extends BaseTransformer {
    override def transform(tree: DependencyTree): DependencyTree = {
      tree.children.find(c => c._2 == "cc" && c._1.token.posTag == "CC" && c._1.children.size == 0) match {
        case None => transformChildren(tree)
        case Some((child, label)) => {
          transformChildren(removeTree(tree, child))
        }
      }
    }
  }

  object RemoveAuxiliaries extends BaseTransformer {
    override def transform(tree: DependencyTree): DependencyTree = {
      tree.children.find(c => c._2 == "aux" && c._1.token.posTag == "MD" && c._1.children.size == 0) match {
        case None => transformChildren(tree)
        case Some((child, label)) => {
          transformChildren(removeTree(tree, child))
        }
      }
    }
  }
}

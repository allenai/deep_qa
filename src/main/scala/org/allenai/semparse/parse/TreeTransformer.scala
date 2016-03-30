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
   * TODO(matt): I may need to change token indices with this, too, as getting the yield of a
   * dependency tree sorts by token index.  So far I only call yield on NPs, which don't get
   * modified with these methods, so we should be ok.  But this could be an issue later.
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
   * Removes the _first_ occurrence of a child with label childLabel.  ThiNOT recursive, and will
   * throw an error if the given child label is not found.
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
   * Finds the subtree that corresponds to a wh-phrase, which would be replaced if we were turning
   * a question into a declarative sentence.  We currently assume that all wh-phrases are of the
   * form "(Which|What) NP? VP", where "which" or "what" is the (determiner) child of the NP, or
   * "Which of NP VP", where "which" is the (still determiner) head of the NP.
   */
  def findWhPhrase(tree: DependencyTree): Option[DependencyTree] = {
    if (isWhWord(tree.token)) {
      // We'll check this as a base case; we'll have already caught cases like "which gas", where
      // "gas" is the head, higher up in the tree.
      Some(tree)
    } else {
      val hasWhChild = tree.children.exists(c => isWhWord(c._1.token) && c._1.children.size == 0)
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

  def isWhWord(token: Token): Boolean = {
    token.lemma == "which" || token.lemma == "what"
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
          val newWord = tree.token.word + "_" + child.token.word
          val newLemma = tree.token.lemma + "_" + child.token.lemma
          val newToken = Token(newWord, tree.token.posTag, newLemma, tree.token.index)
          val newTree = DependencyTree(newToken, tree.children.filterNot(_._2 == "prt"))
          transformChildren(newTree)
        }
      }
    }
  }

  class ReplaceWhPhrase(replaceWith: DependencyTree) extends BaseTransformer {
    override def transform(tree: DependencyTree) = {
      findWhPhrase(tree) match {
        case None => tree
        case Some(whTree) => replaceTree(tree, whTree, replaceWith)
      }
    }
  }
}

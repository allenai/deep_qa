package org.allenai.semparse.parse

trait TreeTransformer {
  def transform(tree: DependencyTree): DependencyTree
}

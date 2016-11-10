package org.allenai.deep_qa.data

import com.mattg.util.FileUtil

import scala.collection.mutable

case class Dataset[T <: Instance](instances: Seq[T]) {
  def writeToFiles(filenames: Seq[String], withIndices: Boolean, fileUtil: FileUtil) {
    val strings = instances.map(_.asStrings)

    // We do this zipWithIndex and a map on the whole list for each one, because scala doesn't have
    // an unzip method for arbitrary-length lists...
    for ((filename, i) <- filenames.zipWithIndex) {
      val fileStrings = strings.map(_(i))
      val indexed = if (withIndices) {
        fileStrings.zipWithIndex.map { case (string, index) => s"$index\t$string" }
      } else {
        fileStrings
      }
      fileUtil.writeLinesToFile(filename, indexed)
    }
  }
}

trait DatasetReader[T <: Instance] {
  def readFile(filename: String): Dataset[T]
}

object DatasetReader {
  val readers = new mutable.HashMap[String, FileUtil => DatasetReader[_]]
  readers.put("babi", (fileUtil) => new BabiDatasetReader(fileUtil))
  readers.put("children's books", (fileUtil) => new ChildrensBookDatasetReader(fileUtil))
  readers.put("snli", (fileUtil) => new SnliDatasetReader(fileUtil))
}

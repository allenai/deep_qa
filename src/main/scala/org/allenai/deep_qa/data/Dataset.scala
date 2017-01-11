package org.allenai.deep_qa.data

import com.mattg.util.FileUtil

import scala.collection.mutable

case class Dataset[T <: Instance](instances: Seq[T]) {
  def writeToFiles(filenames: Seq[String], withIndices: Boolean, fileUtil: FileUtil) {
    // Each Instance returns a Seq[Seq[String]].  This call flattens together all of the inner
    // Seq[String]s, giving one list of Seq[Seq[String]] with all of the instances combined.
    val strings = instances.map(_.asStrings).transpose.map(_.flatten)

    for ((filename, lines) <- filenames.zip(strings)) {
      val indexed = if (withIndices) {
        lines.zipWithIndex.map { case (string, index) => s"$index\t$string" }
      } else {
        lines
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
  readers.put("open qa", (fileUtil) => new OpenQADatasetReader(fileUtil))
  readers.put("squad", (fileUtil) => new SquadDatasetReader(fileUtil))
  readers.put("who did what", (fileUtil) => new WhoDidWhatDatasetReader(fileUtil))
  readers.put("newsqa", (fileUtil) => new NewsQaDatasetReader(fileUtil))
}

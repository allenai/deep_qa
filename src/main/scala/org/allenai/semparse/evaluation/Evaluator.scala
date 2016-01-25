package org.allenai.semparse.evaluation

import org.allenai.semparse.lisp.Environment

import edu.cmu.ml.rtw.users.matt.util.FileUtil

class Evaluator(
  env: Environment,
  queryFile: String,
  dataFile: String,
  fileUtil: FileUtil = new FileUtil
) {
  val entityNames = readEntityNames(dataFile)

  def readEntityNames(dataFile: String) = {
    for (line <- fileUtil.getLineIterator(dataFile)) {
      val fields = line.split("\t")
      val mids = fields(0).split(" ")
      val names = fields(1).trim().split("\" \"")
    }
  }
}

object Evaluator {
}

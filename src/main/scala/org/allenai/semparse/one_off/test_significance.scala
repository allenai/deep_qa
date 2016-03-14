package org.allenai.semparse.one_off

import edu.cmu.ml.rtw.pra.experiments.ExperimentScorer
import com.mattg.util.FileUtil

// TODO(matt): This should be put into a common space, not a one_off script.
object test_significance {
  val combined_ap_file = "/home/mattg/clone/tacl2015-factorization/results/large/combined_ap.txt"
  val distributional_ap_file = "/home/mattg/clone/tacl2015-factorization/results/large/distributional_ap.txt"
  val formal_ap_file = "/home/mattg/clone/tacl2015-factorization/results/large/formal_ap.txt"

  val fileUtil = new FileUtil

  def main(args: Array[String]) {
    val combined_ap = fileUtil.getLineIterator(combined_ap_file).map(_.split(": ")(1).toDouble).toSeq
    val distributional_ap = fileUtil.getLineIterator(distributional_ap_file).map(_.split(": ")(1).toDouble).toSeq
    val formal_ap = fileUtil.getLineIterator(formal_ap_file).map(_.split(": ")(1).toDouble).toSeq
    println("Combined MAP: " + (combined_ap.sum / combined_ap.size))
    println("Distributional MAP: " + (distributional_ap.sum / distributional_ap.size))
    println("Formal MAP: " + (formal_ap.sum / formal_ap.size))
    println("Combined vs. distributional: " + ExperimentScorer.getPValue(combined_ap.zip(distributional_ap)))
    println("Combined vs. formal: " + ExperimentScorer.getPValue(combined_ap.zip(formal_ap)))
    println("Distributional vs. formal: " + ExperimentScorer.getPValue(distributional_ap.zip(formal_ap)))
  }
}

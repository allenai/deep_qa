package org.allenai.semparse.pipeline.science_data

import org.json4s._

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.allenai.semparse.parse.Atom
import org.allenai.semparse.parse.Conjunction
import org.allenai.semparse.parse.Logic
import org.allenai.semparse.parse.Predicate

import scala.util.Random

/**
 * The goal of this class is to take some positive training data as input, then produce a set of
 * corrupted data for each positive instance, so we can train a model using noise contrastive
 * estimation.
 */
class SentenceCorruptor(
  params: JValue,
  fileUtil: FileUtil
) extends Step(Some(params), fileUtil) {
  implicit val formats = DefaultFormats
  override val name = "Sentence Corruptor"

  val validParams = Seq("positive data")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val sentenceProcessor = new ScienceSentenceProcessor(params \ "positive data", fileUtil)
  val positiveDataFile = sentenceProcessor.outputFile
  val dataName = sentenceProcessor.dataName
  val outputFile = s"data/science/$dataName/corrupted_data.tsv"

  val numPartitions = 100

  override val inputs: Set[(String, Option[Step])] = Set((positiveDataFile, Some(sentenceProcessor)))
  override val outputs = Set(outputFile)
  override val paramFile = outputs.head.dropRight(4) + "_params.json"
  override val inProgressFile = outputs.head.dropRight(4) + "_in_progress"

  override def _runStep() {
    val conf = new SparkConf().setAppName(s"Sentence Processor")
      .set("spark.driver.maxResultSize", "0")
      .set("spark.network.timeout", "100000")
      .set("spark.akka.frameSize", "1028")
      .setMaster("local[*]")

    val sc = new SparkContext(conf)

    fileUtil.mkdirsForFile(outputFile)

    corruptData(sc, positiveDataFile, numPartitions, fileUtil)
    sc.stop()
  }

  def corruptData(sc: SparkContext, positiveDataFile: String, numPartitions: Int, fileUtil: FileUtil) {
    val random = new Random()
    val instances = sc.textFile(positiveDataFile, numPartitions).flatMap(SentenceCorruptor.parseToLogic)

    // For this initial, naive approach, we're just going to sample by frequency in the positive
    // data, so we need count up how many times we see each predicate and argument in the data.
    val wordOccurrences = instances.flatMap(SentenceCorruptor.getLogicCounts)

    val predicateOccurrences = wordOccurrences.filter(_._1 == "predicate").map(c => (c._2, c._3))
    val predicateCountsPB = predicateOccurrences.reduceByKey(_ + _).collectAsMap
    val totalPredicateCount = predicateCountsPB.map(_._2).sum
    val predicateCounts = sc.broadcast(predicateCountsPB.toSeq)

    val atomOccurrences = wordOccurrences.filter(_._1 == "atom").map(c => (c._2, c._3))
    val atomCountsPB = atomOccurrences.reduceByKey(_ + _).collectAsMap
    val totalAtomCount = atomCountsPB.map(_._2).sum
    val atomCounts = sc.broadcast(atomCountsPB.toSeq)

    // Now, with the counts, we actually corrupt the data.
    val instancesWithCorruptions = instances.map(
      SentenceCorruptor.corruptInstance(
        random,
        atomCounts.value,
        totalAtomCount,
        predicateCounts.value,
        totalPredicateCount
      )
    )

    // And then output it to a file.
    val lines = instancesWithCorruptions.map(i => {
      i._1.toString + "\t" + i._2.map(_.toString).mkString("\t")
    }).collect()
    fileUtil.writeLinesToFile(outputFile, lines)

    // Cleaning this up is important, as other steps in the pipeline might need to use a spark
    // context.  If we don't stop it, the rest of the pipeline will break.
  }
}

object SentenceCorruptor {

  def parseToLogic(line: String): Option[Logic] = {
    // TODO(matt): this isn't the best format to be passing around.  There's a reason this is
    // currently called the "debug" format...
    line.split(" -> ").last match {
      case "" => None
      case logicalFormString => Some(Logic.fromString(logicalFormString))
    }
  }

  def getLogicCounts(logic: Logic): Seq[(String, String, Int)] = {
    logic match {
      case Atom(atom) => Seq(("atom", atom, 1))
      case Predicate(predicate, arguments) => Seq(("predicate", predicate, 1)) ++ arguments.flatMap(getLogicCounts)
      case Conjunction(arguments) => arguments.toSeq.flatMap(getLogicCounts)
    }
  }

  def getCorruptionLocations(logic: Logic): Seq[(String, String, Logic)] = {
    logic match {
      case a: Atom => { Seq(("atom", a.symbol, a)) }
      case p: Predicate => { Seq(("predicate", p.predicate, p)) ++ p.arguments.flatMap(getCorruptionLocations) }
      case Conjunction(arguments) => arguments.toSeq.flatMap(getCorruptionLocations)
    }
  }

  def corruptInstance(
    random: Random,
    atomCounts: Seq[(String, Int)],
    totalAtomCount: Int,
    predicateCounts: Seq[(String, Int)],
    totalPredicateCount: Int
  )(
    instance: Logic
  ): (Logic, Seq[Logic]) = {
    val corruptionLocations = getCorruptionLocations(instance)
    val locationToCorrupt = corruptionLocations(random.nextInt(corruptionLocations.size))
    val corrupted = if (locationToCorrupt._1 == "atom") {
      corruptLocation(random, atomCounts, totalAtomCount, instance, locationToCorrupt)
    } else {
      corruptLocation(random, predicateCounts, totalPredicateCount, instance, locationToCorrupt)
    }
    (instance, Seq(corrupted))
  }

  def corruptLocation(
    random: Random,
    replacementCount: Seq[(String, Int)],
    replacementTotalCount: Int,
    instance: Logic,
    locationToCorrupt: (String, String, Logic)
  ): Logic = {
    if (instance == locationToCorrupt._3) {
      instance match {
        case Atom(atom) => {
          val newAtom = sampleReplacement(random, replacementCount, replacementTotalCount)
          Atom(newAtom)
        }
        case Predicate(predicate, arguments) => {
          val newPredicate = sampleReplacement(random, replacementCount, replacementTotalCount)
          Predicate(newPredicate, arguments)
        }
        case _ => throw new RuntimeException("Not sure how you got here...")
      }
    } else {
      instance match {
        case a: Atom => { a }
        case Predicate(predicate, arguments) => {
          Predicate(predicate,
            arguments.map(l => corruptLocation(
              random,
              replacementCount,
              replacementTotalCount,
              l,
              locationToCorrupt
              )
            )
          )
        }
        case Conjunction(arguments) => {
          Conjunction(arguments.map(l => corruptLocation(
            random,
            replacementCount,
            replacementTotalCount,
            l,
            locationToCorrupt
            )
          ))
        }
      }
    }
  }

  def sampleReplacement(random: Random, counts: Seq[(String, Int)], totalCount: Int): String = {
    var i = random.nextInt(totalCount)
    val sampled = counts.find(count => {
      i -= count._2
      i < 0
    })
    sampled match {
      case None => counts.last._1
      case Some((string, count)) => string
    }
  }
}

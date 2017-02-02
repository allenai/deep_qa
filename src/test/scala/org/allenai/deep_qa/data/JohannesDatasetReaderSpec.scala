package org.allenai.deep_qa.data

import com.mattg.util.FileUtil
import org.scalatest._

class JohannesDatasetReaderSpec extends FlatSpecLike with Matchers {

  val fileUtil = new FileUtil
  val support1 = "Mesophiles grow best in moderate temperature, typically between 25\u00b0C and " +
    "40\u00b0C (77\u00b0F and 104\u00b0F). Mesophiles are often found living in or on the bodies of " +
    "humans or other animals. The optimal growth temperature of many pathogenic mesophiles is " +
    "37\u00b0C (98\u00b0F), the normal human body temperature. Mesophilic organisms have important " +
    "uses in food preparation, including cheese, yogurt, beer and wine."
  val question1 = "What type of organism is commonly used in preparation of foods such as " +
    "cheese and yogurt?"
  val support2 = "Without Coriolis Effect the global winds would blow north to south or south " +
    "to north. But Coriolis makes them blow northeast to southwest or the reverse in the Northern " +
    "Hemisphere. The winds blow northwest to southeast or the reverse in the southern hemisphere."
  val question2 = "What phenomenon makes global winds blow northeast to southwest or the " +
    "reverse in the northern hemisphere and northwest to southeast or the reverse in the " +
    "southern hemisphere?"

  val datasetFile = "./dataset"
  val datasetFileContents = s"""[
      |                {
      |                                "support": "${support1}",
      |                                "distractor2": "gymnosperms",
      |                                "distractor3": "viruses",
      |                                "question": "${question1}",
      |                                "distractor1": "protozoa",
      |                                "correct_answer": "mesophilic organisms"
      |                },
      |                {
      |                                "support": "${support2}",
      |                                "distractor2": "centrifugal effect",
      |                                "distractor3": "tropical effect",
      |                                "question": "${question2}",
      |                                "distractor1": "muon effect",
      |                                "correct_answer": "coriolis effect"
      |                }
      |]""".stripMargin
  fileUtil.mkdirsForFile(datasetFile)
  fileUtil.writeContentsToFile(datasetFile, datasetFileContents)
  val reader = new JohannesDatasetReader(fileUtil)
  // Note that the label is the index of the correct answer in the sorted sequence of options.
  val options1 = Seq("gymnosperms", "mesophilic organisms", "protozoa", "viruses")
  val options2 = Seq("centrifugal effect", "coriolis effect", "muon effect", "tropical effect")
  "readFile" should "return a correct dataset" in {
    val dataset = reader.readFile(datasetFile)
    fileUtil.deleteFile(datasetFile)

    dataset.instances.size should be(2)
    dataset.instances(0) should be(McQuestionAnswerInstance(support1, question1, options1, Some(1)))
    dataset.instances(1) should be(McQuestionAnswerInstance(support2, question2, options2, Some(1)))
  }
}

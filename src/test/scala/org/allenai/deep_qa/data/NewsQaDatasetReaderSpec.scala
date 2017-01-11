package org.allenai.deep_qa.data

import com.mattg.util.FakeFileUtil
import org.scalatest._

class NewsQaDatasetReaderSpec extends FlatSpecLike with Matchers {

  val fileUtil = new FakeFileUtil
  val header = "question_text,label,answer_string,passage"
  val question1 = "What was the amount of children murdered?"
  val label1 = "288:290"
  val answer1 = "19"
  val passage1 = "NEW DELHI, India (CNN) -- A high court in northern India on Friday acquitted "+
  "a wealthy businessman facing the death sentence for the killing of a teen in a case "+
  "dubbed \"\"the house of horrors.\"\" Moninder Singh Pandher was sentenced to death by a lower "+
  "court in February. The teen was one of 19 victims -- children and young women -- in one of the "+
  "most gruesome serial killings in India in recent years. The Allahabad high court has acquitted "+
  "Moninder Singh Pandher, his lawyer Sikandar B. Kochar told CNN. Pandher and his domestic employee "+
  "Surinder Koli were sentenced to death in February by a lower court for the rape and murder of "+
  "the 14-year-old. The high court upheld Koli's death sentence, Kochar said. The two were arrested "+
  "two years ago after body parts packed in plastic bags were found near their home in Noida, a New "+
  "Delhi suburb. Their home was later dubbed a \"\"house of horrors\"\" by the Indian media. Pandher "+
  "was not named a main suspect by investigators initially, but was summoned as co-accused during "+
  "the trial, Kochar said. Kochar said his client was in Australia when the teen was raped and killed. "+
  "Pandher faces trial in the remaining 18 killings and could remain in custody, the attorney said."
  val fixedPassage1 = passage1.replace("\"\"", "\"")

  val question2 = "Where was one employee killed?"
  val label2 = "34:59"
  val answer2 = "Sudanese region of Darfur"
  val passage2 = "(CNN) -- Fighting in the volatile Sudanese region of Darfur has sparked another "+
  "wave of refugees into Chad and left a Red Cross employee dead, according to international "+
  "agencies. Refugee camps in eastern Chad house about 300,000 people who fled violence in the "+
  "Darfur region of Sudan. The U.N. High Commissioner for Refugees said on Monday that more than "+
  "12,000 people have fled militia attacks over the last few days from Sudan's Darfur region to "+
  "neighboring Chad, still recovering from a recent attempt by rebels there to topple the "+
  "government. \"\"Most of the new arrivals in Chad had already been displaced in Darfur in "+
  "recent years. They are really tired of being attacked and having to move,\"\" said UNHCR's "+
  "Jorge Holly. \"\"All the new refugees we talked to said they did not want to go back to Darfur "+
  "at this point, they wanted to be transferred to a refugee camp in eastern Chad.\"\" This latest "+
  "influx of refugees in Chad aggravates an already deteriorating security situation across this "+
  "politically unstable region of Africa. Before the latest flight into Chad, the UNHCR and its "+
  "partner groups \"\"were taking care of 240,000 Sudanese refugees in 12 camps in eastern Chad and "+
  "some 50,000 from Central African Republic in the south of the country.\"\" Up to 30,000 people "+
  "in Chad fled the country for Cameroon during the rebel-government fighting. E-mail to a friend"
  val fixedPassage2 = passage2.replace("\"\"", "\"")

  val question3 = "who did say South Africa did not issue a visa on time?"
  val label3 = "103:126"
  val answer3 = "Archbishop Desmond Tutu"
  val passage3 = "Johannesburg (CNN) -- Miffed by a visa delay that led the Dalai Lama to cancel "+
  "a trip to South Africa, Archbishop Desmond Tutu lashed out at his government Tuesday, saying it "+
  "had acted worse than apartheid regimes and had forgotten all that the nation stood for. \"\"When "+
  "we used to apply for passports under the apartheid government, we never knew until the last "+
  "moment what their decision was,\"\" Tutu said at a news conference. \"\"Our government is worse than "+
  "the apartheid government because at least you were expecting it from the apartheid "+
  "government. \"\"I have to say that I can't believe this. I really can't believe this,\"\" Tutu "+
  "said. \"\"You have to wake me up and tell me this is actually happening here.\"\" The Dalai "+
  "Lama scrapped his planned trip to South Africa this week after the nation failed to issue "+
  "him a visa in time, his spokesman said. Visa applications for him and his entourage were "+
  "submitted to the South African High Commission in New Delhi, India, at the end of August, "+
  "and original passports were submitted on September 20, more than two weeks ago, a statement on "+
  "his website said."
  val fixedPassage3 = passage3.replace("\"\"", "\"")

  val question4 = "Who is Radu Mazare?"
  val label4 = "191:222"
  val answer4 = "mayor of the town of Constanta,"
  val passage4 = "(CNN) -- Jewish organizations called for a Romanian official to resign and "+
  "face a criminal investigation after he wore a Nazi uniform during a fashion show over the "+
  "weekend. Radu Mazare, the mayor of the town of Constanta, wore a Nazi uniform during a fashion "+
  "show over the weekend. Radu Mazare, the mayor of the town of Constanta, and his 15-year-old "+
  "son \"\"entered the stage marching the clearly identifiable Nazi 'goose step,'\"\" the Center "+
  "for Monitoring and Combating anti-Semitism in Romania said in a letter to the country's "+
  "prosecutor general. The organization's director, Marco Katz, said Mazare had broken Romanian "+
  "law and encouraged his son to do the same, \"\"educating him to treat the law with contempt.\"\" "+
  "Katz said Mazare was sending a message \"\"that to wear Nazi uniforms and to march the Nazi "+
  "steps is legal and 'in vogue' in Romania.\"\" He urged the authorities and the head of Mazare's "+
  "Social Democrat party to show that message \"\"will be strongly countermanded.\"\" Mazare, 41, "+
  "said he had not noticed the Nazi swastika symbol on the uniform before he wore it, according "+
  "to the Romanian Times newspaper. \"\"I checked it before I put it on but the swastika was very "+
  "small and I didn't see it,\"\" he said. \"\"I really liked the look of the uniform after seeing "+
  "it in the Tom Cruise film 'Valkyrie.' I bought it from a costume hire shop in Germany.\"\" A top "+
  "Nazi hunter said Mazare should quit."
  val fixedPassage4 = passage4.replace("\"\"", "\"")

  val datasetFile = "/dataset"
  val datasetFileContents = s"""${header}
      |${question1},${label1},${answer1},"${passage1}"
      |${question2},${label2},${answer2},"${passage2}"
      |${question3},${label3},${answer3},"${passage3}"
      |${question4},${label4},"${answer4}","${passage4}"""".stripMargin

  fileUtil.addFileToBeRead(datasetFile, datasetFileContents)
  val reader = new NewsQaDatasetReader(fileUtil)
  val dataset = reader.readFile(datasetFile)

  "readFile" should "return a correct dataset" in {
    dataset.instances.size should be (4)

    // note that whitespace was trimmed following the answer.
    dataset.instances(0) should be(SpanPredictionInstance(question1, fixedPassage1, Some(288, 290)))
    dataset.instances(1) should be(SpanPredictionInstance(question2, fixedPassage2, Some(34, 59)))
    dataset.instances(2) should be(SpanPredictionInstance(question3, fixedPassage3, Some(103, 126)))
    dataset.instances(3) should be(SpanPredictionInstance(question4, fixedPassage4, Some(191, 222)))

    dataset.instances(0).passage.substring(dataset.instances(0).label.get._1, dataset.instances(0).label.get._2) should be (answer1)
    dataset.instances(1).passage.substring(dataset.instances(1).label.get._1, dataset.instances(1).label.get._2) should be (answer2)
    dataset.instances(2).passage.substring(dataset.instances(2).label.get._1, dataset.instances(2).label.get._2) should be (answer3)
    dataset.instances(3).passage.substring(dataset.instances(3).label.get._1, dataset.instances(3).label.get._2) should be (answer4)
  }
}

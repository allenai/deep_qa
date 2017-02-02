package org.allenai.deep_qa.data

import com.mattg.util.FileUtil
import org.scalatest._

class WhoDidWhatDatasetReaderSpec extends FlatSpecLike with Matchers {

  val fileUtil = new FileUtil
  val rightContext1 = "walked through the general managers meetings in Dana Point , Calif. , " +
    "last week as proud as a peacock ."
  val leftContext1 = ""
  val question1 = s"XXX walked through the general managers meetings in Dana Point , Calif. , " +
    "last week as proud as a peacock ."
  val passage1 = "Joe Maddon has won the American League Manager of the Year after guiding the " +
    "Tampa Bay Rays from baseball's basement to the World Series. Lou Piniella of the Chicago " +
    "Cubs is the National League winner. Maddon succeeded Piniella as Tampa Bay manager in 2006. " +
    "He was an easy winner Wednesday in balloting by the Baseball Writers' Association of " +
    "America. He received all but one of the 28 first-place votes -- the other went to " +
    "Minnesota's Ron Gardenhire. Piniella led the NL Central champion Cubs to the league's " +
    "best record and beat out Charlie Manuel of the World Series champion Philadelphia Phillies to " +
    "earn his third Manager of the Year award and first in the NL."

  val leftContext2 = "President-elect Barack Obama is poised to move swiftly to reverse " +
    "actions that"
  val rightContext2 = "took using executive authority , and his transition team is reviewing " +
    "limits on stem cell research and the expansion of oil and gas drilling , among other " +
    "issues , members of the team said Sunday ."
  val question2 = "President-elect Barack Obama is poised to move swiftly to reverse " +
    "actions that XXX took using executive authority , and his transition team is reviewing " +
    "limits on stem cell research and the expansion of oil and gas drilling , among other " +
    "issues , members of the team said Sunday ."
  val passage2 = "US president-elect Barack Obama is reviewing President George W. Bush's " +
    "executive orders on oil drilling and stem cell research and could reverse them, his " +
    "transition chief said Sunday. Many of the orders are at odds with Democrat Obama's " +
    "approach to the environment and health care. \"I think across the board, on stem " +
    "cell research, on a number of areas, you see the Bush administration, even today, moving " +
    "aggressively to do things that I think are probably not in the interest of the " +
    "country,\" John Podesta told Fox. \"We're looking at -- again, in virtually every agency " +
    "to see where we can move forward, whether that's on energy transformation, on improving " +
    "health care, on stem cell research,\" he said. Podesta, who also served as White House " +
    "chief of staff under president Bill Clinton, said he would not \"preview decisions that " +
    "he (Obama) has yet to make.\" However he pointed out that \"as a candidate, Senator " +
    "Obama said that he wanted all the Bush executive orders reviewed, and decide which ones " +
    "should be kept, and which ones should be repealed, and which ones should be " +
    "amended.\" Among the measures that Podesta raised were the Bush administration's move " +
    "to authorize oil and gas drilling in the western state of Utah, and embryonic stem cell " +
    "research which Bush has limited because he views it as destruction of human life. \"They " +
    "want to have oil and gas drilling in some of the most sensitive, fragile lands in Utah that " +
    "they're going to try to do right as they -- walking out the door. I think that's a " +
    "mistake,\" he said. Podesta signaled that Obama would look to take swift action once he " +
    "takes power on January 20. \"There's a lot that the president can do using his executive " +
    "authority without waiting for congressional action, and I think we'll see the president do " +
    "that to try to restore the -- a sense that the country is working on behalf of the common " +
    "good,\" he said. \"We're going to try to restore wages, give people the right kind of " +
    "ways that they can build on their own lives, and when they work hard that they'll be " +
    "rewarded for it,\" he added. Obama is scheduled to visit the White House Monday for talks " +
    "with Bush on the transition."

  val datasetFile = "./dataset"
  val datasetFileContents = s"""<?xml version="1.0" encoding="UTF-8"?>
      |<ROOT>
      |<mc>
      | <question id="NYT_ENG_20081109.0068">
      |  <leftcontext>${leftContext1}</leftcontext>
      |  <leftblank>
      |   Charlie `` The Manager ''
      |  </leftblank>
      |  <blank type="PERSON">
      |   Manuel
      |  </blank>
      |  <rightblank></rightblank>
      |  <rightcontext>
      |   ${rightContext1}
      |  </rightcontext>
      | </question>
      | <contextart id="APW_ENG_20081112.1165">
      |  ${passage1}
      | </contextart>
      | <choice idx="0" correct="true">
      |  Joe Maddon
      | </choice>
      | <choice idx="1" correct="false">
      |  Lou Piniella
      | </choice>
      | <choice idx="2" correct="false">
      |  Ron Gardenhire
      | </choice>
      | <choice idx="3" correct="false">
      |  Charlie Manuel
      | </choice>
      |</mc>
      |<mc>
      | <question id="NYT_ENG_20081109.0135">
      |  <leftcontext>
      |   ${leftContext2}
      |  </leftcontext>
      |  <leftblank>
      |   President
      |  </leftblank>
      |  <blank type="PERSON">
      |   Bush
      |  </blank>
      |  <rightblank></rightblank>
      |  <rightcontext>
      |   ${rightContext2}
      |  </rightcontext>
      | </question>
      | <contextart id="AFP_ENG_20081109.0073">
      |  ${passage2}
      | </contextart>
      | <choice idx="0" correct="true">
      |  George W. Bush
      | </choice>
      | <choice idx="1" correct="false">
      |  John Podesta
      | </choice>
      | <choice idx="2" correct="false">
      |  Fox
      | </choice>
      | <choice idx="3" correct="false">
      |  Bill Clinton
      | </choice>
      |</mc>
      |</ROOT>""".stripMargin
  fileUtil.mkdirsForFile(datasetFile)
  fileUtil.writeContentsToFile(datasetFile, datasetFileContents)
  val reader = new WhoDidWhatDatasetReader(fileUtil)
  val answers1 = Seq("Joe Maddon", "Lou Piniella", "Ron Gardenhire", "Charlie Manuel")
  val answers2 = Seq("George W. Bush", "John Podesta", "Fox", "Bill Clinton")

  "readFile" should "return a correct dataset" in {
    val dataset = reader.readFile(datasetFile)
    fileUtil.deleteFile(datasetFile)
    fileUtil.deleteFile(datasetFile + ".bak")

    dataset.instances.size should be(2)
    dataset.instances(0) should be(McQuestionAnswerInstance(passage1, question1, answers1, Some(0)))
    dataset.instances(1) should be(McQuestionAnswerInstance(passage2, question2, answers2, Some(0)))
  }
}

package org.allenai.deep_qa.solver

import scala.sys.process.Process

import org.allenai.deep_qa.data.Instance
import org.allenai.deep_qa.data.BackgroundInstance
import org.allenai.deep_qa.data.QuestionAnswerInstance
import org.allenai.deep_qa.data.MultipleTrueFalseInstance
import org.allenai.deep_qa.data.TrueFalseInstance
import org.allenai.deep_qa.message.SolverServiceGrpc.SolverServiceBlockingStub
import org.allenai.deep_qa.message.{Instance => MessageInstance}
import org.allenai.deep_qa.message.InstanceType
import org.allenai.deep_qa.message.SolverServiceGrpc
import org.allenai.deep_qa.message.QuestionRequest
import org.allenai.deep_qa.message.QuestionResponse

import com.mattg.util.FileUtil
import com.typesafe.scalalogging.LazyLogging

import io.grpc.ManagedChannel
import io.grpc.okhttp.OkHttpChannelBuilder

import java.io.File
import java.util.concurrent.TimeUnit

import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods.{parse,pretty,render}

/**
 * This Client connects to a python server that returns predictions from a trained model.  The only
 * thing we do here is convert between an Instance object and an Instance proto, then send the
 * proto off to the server, returning the scores we get back.
 */
class Client(serverDirectory: File, modelParams: JValue) extends LazyLogging {
  val host = "localhost"
  val port = 50051

  val serverStartTime = 10000
  val fileUtil = new FileUtil
  val params: JValue = ("server" -> ("port" -> port)) ~ ("solver" -> modelParams)
  val modelParamFile = serverDirectory.getAbsolutePath() + "/tmp_model_param_file.json"
  logger.info("Writing model params to disk")
  fileUtil.writeContentsToFile(modelParamFile, pretty(render(params)))
  logger.info("Starting the server process")
  val serverScript = "src/main/python/server.py"
  val serverProcess = Process(s"""python ${serverScript} ${modelParamFile}""", serverDirectory).run
  logger.info(s"Sleeping ${serverStartTime / 1000.0} seconds to give the server time to boot")
  Thread.sleep(serverStartTime)  // we'll give the server a bit of time to boot up before trying to connect.
  logger.info("Connecting to the server")
  val channel = OkHttpChannelBuilder.forAddress(host, port).usePlaintext(true).build
  val blockingStub = SolverServiceGrpc.blockingStub(channel)

  def shutdown(): Unit = {
    logger.info("Closing connection to the server")
    channel.shutdown.awaitTermination(5, TimeUnit.SECONDS)
    logger.info("Shutting down the server")
    serverProcess.destroy()
    logger.info("Deleting model param file")
    fileUtil.deleteFile(modelParamFile)
  }

  def answerQuestion(instance: Instance): Seq[Double] = {
    val response = sendMessage(instanceToMessage(instance))
    response.scores
  }

  def instanceToMessage(instance: Instance): MessageInstance = {
    instance match {
      case i: BackgroundInstance[_] => {
        val containedMessage = instanceToMessage(i.containedInstance)
        val instanceType = containedMessage.`type`
        val questionText = containedMessage.question
        val answerOptions = containedMessage.answerOptions
        val background = i.background
        MessageInstance(instanceType, questionText, answerOptions, background, Seq())
      }
      case i: QuestionAnswerInstance => {
        val instanceType = InstanceType.QUESTION_ANSWER
        val questionText = i.question
        val answerOptions = i.answers
        MessageInstance(instanceType, questionText, answerOptions, Seq(), Seq())
      }
      case i: MultipleTrueFalseInstance[_] => {
        val instanceType = InstanceType.MULTIPLE_TRUE_FALSE
        val containedInstances = i.instances.map(instanceToMessage)
        MessageInstance(instanceType, "", Seq(), Seq(), containedInstances)
      }
      case i: TrueFalseInstance => {
        val instanceType = InstanceType.TRUE_FALSE
        val questionText = i.statement
        val answerOptions = Seq()
        MessageInstance(instanceType, questionText, answerOptions, Seq())
      }
    }
  }

  def sendMessage(message: MessageInstance): QuestionResponse = {
    val request: QuestionRequest = QuestionRequest(Some(message))
    blockingStub.answerQuestion(request)
  }
}

/**
 * This is just for testing things, and showing how to instantiate a Client object and have it
 * answer questions.
 */
object Client {

  def main(args: Array[String]) {
    if (args.length != 1) {
      println("USAGE: Client.scala [model_param_file]")
      System.exit(-1)
    }
    val fileUtil = new FileUtil
    val paramFile = args(0)
    val modelParams = parse(fileUtil.readFileContents(paramFile))

    val client: Client = new Client(new File("./"), modelParams)

    val instance = MultipleTrueFalseInstance(Seq(
      BackgroundInstance(TrueFalseInstance("statement 1", None), Seq("background 1")),
      BackgroundInstance(TrueFalseInstance("statement 2", None), Seq("background 2")),
      BackgroundInstance(TrueFalseInstance("statement 3", None), Seq("background 3")),
      BackgroundInstance(TrueFalseInstance("statement 4", None), Seq("background 4"))
    ), None)

    val scores = client.answerQuestion(instance)
    println(s"Scores: [${scores.mkString(" ")}]")
    client.shutdown()
  }
}

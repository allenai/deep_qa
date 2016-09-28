package org.allenai.dlfa.solver

import org.allenai.dlfa.data.
import org.allenai.dlfa.message.SolverServiceGrpc.SolverServiceBlockingStub
import org.allenai.dlfa.message.{Instance => MessageInstance}
import org.allenai.dlfa.message.InstanceType
import org.allenai.dlfa.message.SolverServiceGrpc
import org.allenai.dlfa.message.QuestionRequest
import org.allenai.dlfa.message.QuestionResponse

import io.grpc.ManagedChannel
import io.grpc.okhttp.OkHttpChannelBuilder

import java.util.concurrent.TimeUnit

object Client {

  val config = com.typesafe.config.ConfigFactory.load()

  val host = config.getString("grpc.dlfa.server")
  val port = config.getInt("grpc.dlfa.port")

  def show(response: QuestionResponse) = {
    println(s"Response: [${response.scores.mkString(" ")}]")
  }

  def main(args: Array[String]) {
    val channel: ManagedChannel =
      OkHttpChannelBuilder.forAddress(host, port).usePlaintext(true).build
    val blockingStub = SolverServiceGrpc.blockingStub(channel)
    val client: Client = new Client(channel, blockingStub)

    show(client.compute(Array("testing", "testing2")))
  }
}

class Client(
  private val channel: ManagedChannel,
  private val blockingStub: SolverServiceBlockingStub
) {
  def shutdown(): Unit = {
    channel.shutdown.awaitTermination(5, TimeUnit.SECONDS)
  }

  def compute(
    instanceType: InstanceType,
    questionText: String,
    answerOptions: Seq[String],
    background: Seq[String]
  ) = {
    val instance = MessageInstance(instanceType, questionText, answerOptions, background)
    val request: QuestionRequest = QuestionRequest(Some(instance))
    blockingStub.answerQuestion(request)
  }
}

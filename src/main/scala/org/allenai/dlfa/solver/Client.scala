package org.allenai.dlfa.solver

import org.allenai.dlfa.message.SolverServiceGrpc.SolverServiceBlockingStub
import org.allenai.dlfa.message.{ Instance, SolverServiceGrpc, QuestionRequest, QuestionResponse }

import io.grpc.{ ManagedChannel, ManagedChannelBuilder }

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
      ManagedChannelBuilder.forAddress(host, port).usePlaintext(true).build
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

  def compute(fields: Array[String]) = {
    val request: QuestionRequest = QuestionRequest(Some(Instance(fields)))
    blockingStub.answerQuestion(request)
  }
}

from concurrent import futures
import time

import grpc
from pyhocon import ConfigFactory

from proto import message_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class SolverServer(message_pb2.SolverServiceServicer):
    def __init__(self):
        # TODO(matt): load the model
        pass

    def AnswerQuestion(self, request, context):
        response = message_pb2.QuestionResponse()
        response.scores = [.2, .3, .1, .3]
        return response


def serve():
    # read in the Typesafe-style config file and lookup the port to run on.
    conf = ConfigFactory.parse_file('src/main/resources/application.conf')
    port = conf["grpc.dlfa.port"]

    # create the server and add our RPC "servicer" to it
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    message_pb2.add_SolverServiceServicer_to_server(SolverServer(), server)

    # start the server on the specified port
    server.add_insecure_port('[::]:{0}'.format(port))
    print("starting server")
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()

from concurrent import futures
import random
import sys
import time

import grpc
import numpy
from pyhocon import ConfigFactory

# These have to be before we do any import from keras.  It would be nice to be able to pass in a
# value for this, but that makes argument passing a whole lot more complicated.  If/when we change
# how arguments work (e.g., loading a file), then we can think about setting this as a parameter.
random.seed(13370)
numpy.random.seed(1337)  # pylint: disable=no-member

# pylint: disable=wrong-import-position
from deep_qa.common.checks import ensure_pythonhashseed_set
from deep_qa.common.params import get_choice
from deep_qa.solvers import concrete_solvers

from deep_qa.data.instances.true_false_instance import TrueFalseInstance
from deep_qa.data.instances.multiple_true_false_instance import MultipleTrueFalseInstance
from deep_qa.data.instances.question_answer_instance import QuestionAnswerInstance
from deep_qa.data.instances.background_instance import BackgroundInstance
from proto import message_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class SolverServer(message_pb2.SolverServiceServicer):
    def __init__(self, solver):
        self.solver = solver
        self.solver.load_model()

    # The name of this method is specified in message.proto.
    def AnswerQuestion(self, request, context):
        instance = self.read_instance_message(request.question)
        try:
            scores = self.solver.score_instance(instance)
        except:
            print("Instance was: " + str(instance))
            raise
        response = message_pb2.QuestionResponse()
        for score in scores.tolist():
            response.scores.extend(score)
        return response

    def read_instance_message(self, instance_message):
        # pylint: disable=redefined-variable-type
        instance_type = instance_message.type
        if instance_type == message_pb2.TRUE_FALSE:
            text = instance_message.question
            instance = TrueFalseInstance(text, None, None, self.solver.tokenizer)
        elif instance_type == message_pb2.MULTIPLE_TRUE_FALSE:
            options = []
            for instance in instance_message.contained_instances:
                options.append(self.read_instance_message(instance))
            instance = MultipleTrueFalseInstance(options)
        elif instance_type == message_pb2.QUESTION_ANSWER:
            question = instance_message.question
            options = instance_message.answer_options
            instance = QuestionAnswerInstance(question, options, None, None, self.solver.tokenizer)
        else:
            raise RuntimeError("Unrecognized instance type: " + instance_type)
        if instance_message.background:
            background = instance_message.background
            instance = BackgroundInstance(instance, background)
        return instance


def serve(config_file: str):
    # read in the Typesafe-style config file and lookup the port to run on.
    params = ConfigFactory.parse_file(config_file)
    server_params = params['server']
    port = server_params['port']

    # create the server and add our RPC "servicer" to it
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    solver_params = params['solver']
    model_type = get_choice(solver_params, 'solver_class', concrete_solvers.keys())
    solver_class = concrete_solvers[model_type]
    solver = solver_class(solver_params)
    message_pb2.add_SolverServiceServicer_to_server(SolverServer(solver), server)

    # start the server on the specified port
    server.add_insecure_port('[::]:{0}'.format(port))
    print("starting server")
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


def main():
    ensure_pythonhashseed_set()
    if len(sys.argv) != 2:
        print('USAGE: server.py [config_file]')
        sys.exit(-1)
    config_file = sys.argv[1]
    serve(config_file)


if __name__ == '__main__':
    main()

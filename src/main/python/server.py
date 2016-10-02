from concurrent import futures
import random
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
from deep_qa.solvers import concrete_solvers

from deep_qa.data.text_instance import TrueFalseInstance
from deep_qa.data.text_instance import MultipleChoiceInstance
from deep_qa.data.text_instance import QuestionAnswerInstance
from deep_qa.data.text_instance import BackgroundInstance
from proto import message_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class SolverServer(message_pb2.SolverServiceServicer):
    def __init__(self, solver):
        self.solver = solver
        self.solver.load_model()

    # The name of this method is specified in message.proto.
    def AnswerQuestion(self, request, context):
        instance = self.read_instance_message(request.question)
        scores = self.solver.score_instance(instance)
        response = message_pb2.QuestionResponse()
        for score in scores.tolist():
            response.scores.extend(score)
        return response

    def read_instance_message(self, instance_message):
        print("Reading message:", instance_message)
        # pylint: disable=redefined-variable-type
        instance_type = instance_message.type
        if instance_type == message_pb2.TRUE_FALSE:
            text = instance_message.question
            instance = TrueFalseInstance(text, None, None, self.solver.tokenizer)
        elif instance_type == message_pb2.MULTIPLE_TRUE_FALSE:
            options = []
            for instance in instance_message.contained_instances:
                options.append(self.read_instance_message(instance))
            instance = MultipleChoiceInstance(options)
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


def serve():
    # read in the Typesafe-style config file and lookup the port to run on.
    conf = ConfigFactory.parse_file('src/main/resources/application.conf')
    port = conf["grpc.deep_qa.port"]
    model_type = conf["grpc.deep_qa.model_class"]

    # create the server and add our RPC "servicer" to it
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # TODO(matt): read this from a file somewhere, ideally the same conf file as above.
    params = {
            'model_serialization_prefix': 'models/example',
            'encoder': {'type': 'bow'},
            'knowledge_selector': {'type': 'dot_product'},
            'memory_updater': {'type': 'sum'},
            'entailment_input_combiner': {'type': 'memory_only'},
            'num_memory_layers': 1,
            'max_sentence_length': 125,
            }

    solver_class = concrete_solvers[model_type]
    solver = solver_class(params)
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


if __name__ == '__main__':
    ensure_pythonhashseed_set()
    serve()

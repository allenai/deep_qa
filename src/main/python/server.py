from concurrent import futures
import argparse
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
from dlfa.solvers import concrete_solvers

from dlfa.data.text_instance import TrueFalseInstance
from dlfa.data.text_instance import MultipleChoiceInstance
from dlfa.data.text_instance import QuestionAnswerInstance
from dlfa.data.text_instance import BackgroundInstance
from proto import message_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class SolverServer(message_pb2.SolverServiceServicer):
    def __init__(self, solver_class, args):
        # TODO(matt): having to go through argparse is pretty ugly.  We need to make a better
        # system for specifying arguments, likely using a json or hocon library.
        argparser = argparse.ArgumentParser()
        solver_class.update_arg_parser(argparser)
        argument_list = []
        for key, value in args.items():
            argument_list.append('--' + key)
            if value is not None:
                argument_list.append(value)
        args = argparser.parse_args(argument_list)
        self.solver = solver_class(**vars(args))
        self.solver.load_model(0)

    def AnswerQuestion(self, request, context):
        instance = self.read_instance_message(request.instance)
        scores = self.solver.score_instance(instance)
        response = message_pb2.QuestionResponse()
        response.scores.extend(scores)
        return response

    def read_instance_message(self, instance_message):
        # pylint: disable=redefined-variable-type
        instance_type = instance_message.type
        if instance_type == message_pb2.TRUE_FALSE:
            text = instance_message.question
            instance = TrueFalseInstance(text, None, None, self.solver.tokenizer)
        elif instance_type == message_pb2.MULTIPLE_TRUE_FALSE:
            options = []
            for option in instance_message.answer_options:
                options.append(TrueFalseInstance(option, None, None, self.solver.tokenizer))
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
    port = conf["grpc.dlfa.port"]
    model_type = conf["grpc.dlfa.model_class"]

    # create the server and add our RPC "servicer" to it
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    args = {
            'model_serialization_prefix': 'models/testing',
            'encoder': 'bow',
            'knowledge_selector': 'dot_product',
            'memory_updater': 'sum',
            'entailment_input_combiner': 'memory_only',
            'num_memory_layers': '1',
            'max_sentence_length': '125',
            }
    solver_class = concrete_solvers[model_type]
    message_pb2.add_SolverServiceServicer_to_server(SolverServer(solver_class, args), server)

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

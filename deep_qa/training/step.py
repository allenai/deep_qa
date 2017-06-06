
from typing import List

import tensorflow
import numpy
import keras.backend as K


class Step:
    """
    Runs a computation graph.

    Parameters
    ----------
    inputs: Feed placeholders to the computation graph.
    outputs: Output tensors to fetch.
    updates: Additional update ops to be run at function call.
    """
    def __init__(self,
                 inputs: List,
                 outputs: List,
                 global_step: tensorflow.Variable,
                 summary_writer: tensorflow.summary.FileWriter=None,
                 summary_frequency: int=10,
                 updates=None):

        updates = updates or []
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` to a TensorFlow backend function '
                            'should be a list or tuple.')
        if not isinstance(outputs, (list, tuple)):
            raise TypeError('`outputs` of a TensorFlow backend function '
                            'should be a list or tuple.')
        if not isinstance(updates, (list, tuple)):
            raise TypeError('`updates` in a TensorFlow backend function '
                            'should be a list or tuple.')
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.summary_writer = summary_writer
        self.summary_frequency = summary_frequency
        self.global_step = global_step

        self.summary_operation = tensorflow.summary.merge_all()

        with tensorflow.control_dependencies(self.outputs):
            updates_ops = []
            for update in updates:
                if isinstance(update, tuple):
                    variable, new_value = update
                    updates_ops.append(tensorflow.assign(variable, new_value))
                else:
                    # assumed already an op
                    updates_ops.append(update)
            self.updates_op = tensorflow.group(*updates_ops)

    def __call__(self, inputs):

        current_step = K.eval(self.global_step)
        run_summary = ((self.summary_frequency > 0)
                       and (current_step % self.summary_frequency == 0)
                       and (self.summary_writer is not None))

        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` should be a list or tuple.')
        feed_dict = {}
        for tensor, value in zip(self.inputs, inputs):
            if K.is_sparse(tensor):
                sparse_coo = value.tocoo()
                indices = numpy.concatenate((numpy.expand_dims(sparse_coo.row, 1),
                                             numpy.expand_dims(sparse_coo.col, 1)), 1)
                value = (indices, sparse_coo.data, sparse_coo.shape)
            feed_dict[tensor] = value

        fetches = self.outputs + [self.updates_op]
        if run_summary:
            fetches += [self.summary_operation]

        session = K.get_session()
        returned_fetches = session.run(fetches, feed_dict=feed_dict)
        if run_summary:
            self.summary_writer.add_summary(returned_fetches[-1], current_step)
            self.summary_writer.flush()

        return returned_fetches[:len(self.outputs)]

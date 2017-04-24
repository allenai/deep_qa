# pylint: disable=no-self-use,invalid-name
import keras.backend as K
import tensorflow as tf
from deep_qa.models.memory_networks import MemoryNetwork
from deep_qa.common.params import Params
from ...common.test_case import DeepQaTestCase

class TestAdaptiveMemoryNetwork(DeepQaTestCase):
    # pylint: disable=protected-access
    def test_train_does_not_crash(self):
        self.write_memory_network_files()
        args = Params({'recurrence_mode': {'type': 'adaptive'}, 'knowledge_selector': {'type': 'parameterized'}})
        model = self.get_model(MemoryNetwork, args)
        model.train()

    def test_tf_and_keras_optimise_identical_variables(self):
        self.write_memory_network_files()
        # Make sure that the variables designated as trainable by tensorflow
        # are the same as those designated as trainable by Keras. These could
        # not be equal, for instance, if we manage to use a Keras layer without
        # calling layer.build(). This is very hard to do, but because this solver
        # mixes in some tensorflow, which does not have a "build" equivalent
        # (in that it will happily train a variable which is in a layer which
        # hasn't been built) we check this.
        # Create a new tf session to avoid variables created in other tests affecting this.
        K.clear_session()
        # Add in a layer which is within the adaptive memory step which actually has
        # parameters.
        args = Params({
                'recurrence_mode': {'type': 'adaptive'},
                'knowledge_selector': {'type': 'parameterized'}
        })
        solver = self.get_model(MemoryNetwork, args)
        solver.training_dataset = solver.load_dataset_from_files(solver.train_files)
        solver.set_model_state_from_dataset(solver.training_dataset)
        indexed_dataset = solver.training_dataset.to_indexed_dataset(solver.data_indexer)
        solver.set_model_state_from_indexed_dataset(indexed_dataset)
        model = solver._build_model()
        tf_trainable_variables = tf.trainable_variables()
        keras_trainable_variables = model.trainable_weights
        assert [x.name for x in tf_trainable_variables] == [x.name for x in keras_trainable_variables]

# pylint: disable=no-self-use,invalid-name
from copy import deepcopy

import keras.backend as K
import tensorflow

from deep_qa.training.multi_gpu import pin_variable_device_scope
from deep_qa.common.params import Params
from deep_qa.models.text_classification import ClassificationModel
from ..common.test_case import DeepQaTestCase


class TestMultiGpu(DeepQaTestCase):

    def setUp(self):
        super(TestMultiGpu, self).setUp()
        self.write_true_false_model_files()
        self.args = Params({
                'num_gpus': 2,
        })

    def test_model_can_train_and_load(self):
        self.ensure_model_trains_and_loads(ClassificationModel, self.args)

    def test_pinned_scope_correctly_allocates_ops(self):
        scope_function = pin_variable_device_scope(device="/gpu:0", variable_device="/cpu:0")

        # Should have a cpu scope.
        variable = tensorflow.Variable([])
        # Should have a gpu scope.
        add_op = tensorflow.add(variable, 1.0)

        assert scope_function(variable.op) == "/cpu:0"
        assert scope_function(add_op.op) == "/gpu:0"  # pylint: disable=no-member

    def test_variables_live_on_cpu(self):
        model = self.get_model(ClassificationModel, self.args)
        model.train()

        trainable_variables = model.model.trainable_weights
        for variable in trainable_variables:
            # This is an odd quirk of tensorflow - the devices are actually named
            # slightly differently from their scopes ... (i.e != "/cpu:0")
            assert variable.device == "/device:CPU:0" or variable.device == ""

    def test_multi_gpu_shares_variables(self):
        multi_gpu_model = self.get_model(ClassificationModel, self.args)

        single_gpu_args = deepcopy(self.args)
        single_gpu_args["num_gpus"] = 1
        single_gpu_model = self.get_model(ClassificationModel, single_gpu_args)

        multi_gpu_model.train()
        multi_gpu_variables = [x.name for x in multi_gpu_model.model.trainable_weights]

        K.clear_session()
        single_gpu_model.train()
        single_gpu_variables = [x.name for x in single_gpu_model.model.trainable_weights]

        assert single_gpu_variables == multi_gpu_variables

import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Input

from pyment.interpretability import input_gradients, integrated_gradients, \
                                   saliency_map
from pyment.models.model import Model
from pyment.models.model_type import ModelType


class TwoOutputModel(Model):
    @property
    def type(self):
        return ModelType.REGRESSION


def _model():
    inputs = Input((2, 2, 2), name='inputs')
    x = Flatten()(inputs)
    outputs = Dense(2, use_bias=False, kernel_initializer='ones')(x)

    return TwoOutputModel(inputs, outputs)


def test_input_gradients_keeps_input_shape_for_single_sample():
    model = _model()
    image = np.ones((2, 2, 2), dtype=np.float32)

    gradients = input_gradients(model, image, target_index=0)

    assert gradients.shape == (1, 2, 2, 2)
    assert np.all(gradients == 1)


def test_saliency_map_normalizes_per_sample():
    model = _model()
    images = np.ones((2, 2, 2, 2), dtype=np.float32)

    saliency = saliency_map(model, images, target_index=1)

    assert saliency.shape == images.shape
    assert np.all(saliency == 1)


def test_integrated_gradients_uses_zero_baseline_by_default():
    model = _model()
    image = np.ones((2, 2, 2), dtype=np.float32)

    importance = integrated_gradients(model, image, target_index=0, steps=4)

    assert importance.shape == (1, 2, 2, 2)
    assert np.allclose(importance, 1)

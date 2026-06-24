import numpy as np
import tensorflow as tf

from pyment.models.model import Model

from typing import Optional, Sequence, Union


ArrayLike = Union[np.ndarray, tf.Tensor]


def _as_batched_tensor(inputs: ArrayLike, model: Model) -> tf.Tensor:
    """Return ``inputs`` as a float32 tensor with a batch dimension."""
    tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
    expected_rank = len(model.inputs[0].shape)

    if tensor.shape.rank == expected_rank - 1:
        tensor = tf.expand_dims(tensor, axis=0)

    if tensor.shape.rank != expected_rank:
        raise ValueError((
            f'Expected input rank {expected_rank - 1} (single sample) or '
            f'{expected_rank} (batch), got rank {tensor.shape.rank}'
        ))

    return tensor


def _select_output(outputs: Union[tf.Tensor, Sequence[tf.Tensor]],
                   target_output: Optional[int] = None) -> tf.Tensor:
    """Select a model output tensor from single- or multi-output models."""
    if isinstance(outputs, (list, tuple)):
        if target_output is None:
            if len(outputs) != 1:
                raise ValueError(('target_output is required for models with '
                                  'multiple outputs'))
            target_output = 0

        return outputs[target_output]

    if target_output is not None:
        raise ValueError('target_output can only be used with multi-output models')

    return outputs


def _select_target(predictions: tf.Tensor,
                   target_index: Optional[Union[int, Sequence[int]]] = None
                   ) -> tf.Tensor:
    """Select the scalar prediction(s) whose input gradient is explained."""
    if target_index is None:
        return predictions

    if isinstance(target_index, int):
        return predictions[..., target_index]

    return predictions[(slice(None),) + tuple(target_index)]


def input_gradients(model: Model, inputs: ArrayLike, *,
                    target_index: Optional[Union[int, Sequence[int]]] = None,
                    target_output: Optional[int] = None,
                    absolute: bool = True) -> np.ndarray:
    """Compute gradient-based feature importance for 3D SFCN inputs.

    The returned map has the same shape as the batched model input. For a
    single-output regression model this is the gradient of the prediction with
    respect to each voxel. For classification or multi-output models, use
    ``target_index`` and/or ``target_output`` to choose which score to explain.
    """
    batched_inputs = _as_batched_tensor(inputs, model)

    with tf.GradientTape() as tape:
        tape.watch(batched_inputs)
        outputs = model(batched_inputs, training=False)
        predictions = _select_output(outputs, target_output=target_output)
        target = _select_target(predictions, target_index=target_index)
        score = tf.reduce_sum(target)

    gradients = tape.gradient(score, batched_inputs)

    if gradients is None:
        raise ValueError('Unable to compute gradients for the selected target')

    if absolute:
        gradients = tf.abs(gradients)

    return gradients.numpy()


def saliency_map(model: Model, inputs: ArrayLike, *,
                 target_index: Optional[Union[int, Sequence[int]]] = None,
                 target_output: Optional[int] = None,
                 normalize: bool = True) -> np.ndarray:
    """Compute an absolute input-gradient saliency map.

    When ``normalize`` is true, each sample is scaled independently to the
    range [0, 1], making maps easier to visualize and compare.
    """
    saliency = input_gradients(model, inputs, target_index=target_index,
                               target_output=target_output, absolute=True)

    if normalize:
        axes = tuple(range(1, saliency.ndim))
        max_values = np.max(saliency, axis=axes, keepdims=True)
        saliency = np.divide(saliency, max_values,
                             out=np.zeros_like(saliency),
                             where=max_values != 0)

    return saliency


def integrated_gradients(model: Model, inputs: ArrayLike, *,
                         baseline: ArrayLike = None,
                         target_index: Optional[Union[int, Sequence[int]]] = None,
                         target_output: Optional[int] = None,
                         steps: int = 50,
                         absolute: bool = True) -> np.ndarray:
    """Compute integrated gradients feature importance.

    Integrated gradients average input gradients along a straight path from a
    baseline image (zeros by default) to the input image, reducing noise compared
    with a single saliency pass.
    """
    if steps <= 0:
        raise ValueError('steps must be greater than zero')

    batched_inputs = _as_batched_tensor(inputs, model)

    if baseline is None:
        baseline_tensor = tf.zeros_like(batched_inputs)
    else:
        baseline_tensor = _as_batched_tensor(baseline, model)
        baseline_tensor = tf.broadcast_to(baseline_tensor,
                                          tf.shape(batched_inputs))

    delta = batched_inputs - baseline_tensor
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    total_gradients = tf.zeros_like(batched_inputs)

    for alpha in alphas:
        interpolated = baseline_tensor + alpha * delta
        total_gradients += tf.convert_to_tensor(
            input_gradients(model, interpolated, target_index=target_index,
                            target_output=target_output, absolute=False),
            dtype=tf.float32)

    attributions = delta * total_gradients / tf.cast(steps + 1, tf.float32)

    if absolute:
        attributions = tf.abs(attributions)

    return attributions.numpy()

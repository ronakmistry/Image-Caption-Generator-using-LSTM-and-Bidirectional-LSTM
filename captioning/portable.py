"""Equivalent decoder operations that avoid problematic Metal fused kernels."""
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package='captioning')
class PortableDense(keras.layers.Dense):
    def call(self, inputs, training=None):
        # Keras ops.add may select BiasAdd and trigger Metal activation fusion.
        result = tf.linalg.matmul(inputs, self.kernel)
        if self.bias is not None:
            result = tf.raw_ops.AddV2(x=result, y=self.bias)
        return self.activation(result) if self.activation is not None else result


@keras.saving.register_keras_serializable(package='captioning')
class GatherLSTM(keras.layers.LSTM):
    """Use sequence outputs explicitly; avoid Metal's mismatched final-state output."""
    def __init__(self, *args, return_sequences=False, **kwargs):
        if kwargs.get('go_backwards', False) or kwargs.get('return_state', False):
            raise ValueError('GatherLSTM supports forward caption prefixes without returned states only')
        self.requested_sequences = return_sequences
        kwargs.pop('return_state', None)
        super().__init__(*args, return_sequences=True, return_state=False, **kwargs)

    def call(self, sequences, initial_state=None, mask=None, training=False):
        values = super().call(sequences, initial_state=initial_state, mask=mask, training=training)
        if mask is None:
            last = values[:, -1, :]
        else:
            lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
            indices = tf.stack([tf.range(tf.shape(values)[0]), tf.maximum(lengths - 1, 0)], axis=1)
            last = tf.gather_nd(values, indices)
            values = tf.where(mask[..., None], values, last[:, None, :])
        return values if self.requested_sequences else last

    def compute_output_shape(self, sequences_shape, initial_state_shape=None):
        shape = (sequences_shape[0], sequences_shape[1], self.units)
        return shape if self.requested_sequences else (sequences_shape[0], self.units)

    def compute_mask(self, inputs, mask):
        return mask if self.requested_sequences else None

    def get_config(self):
        config = super().get_config()
        config['return_sequences'] = self.requested_sequences
        return config


def gathered_clone(model):
    def clone(layer):
        config = layer.get_config()
        if isinstance(layer, keras.layers.Dense):
            return PortableDense.from_config(config)
        if isinstance(layer, keras.layers.LSTM):
            return GatherLSTM.from_config(config)
        if isinstance(layer, keras.layers.TimeDistributed):
            return keras.layers.TimeDistributed(clone(layer.layer), name=layer.name)
        return layer.__class__.from_config(config)
    result = keras.models.clone_model(model, clone_function=clone)
    result.set_weights(model.get_weights())
    return result

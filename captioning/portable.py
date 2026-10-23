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


@keras.saving.register_keras_serializable(package='captioning')
class PortableBidirectional(keras.layers.Layer):
    """Masked BLSTM using two forward kernels on valid, reversed prefixes."""
    def __init__(self, units, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.supports_masking = True
        self.forward_layer = GatherLSTM(units, return_sequences=return_sequences)
        self.backward_layer = GatherLSTM(units, return_sequences=return_sequences)

    def build(self, shape):
        self.forward_layer.build(shape)
        self.backward_layer.build(shape)
        super().build(shape)

    def call(self, inputs, mask=None, training=False):
        lengths = (tf.reduce_sum(tf.cast(mask, tf.int32), axis=1) if mask is not None
                   else tf.fill([tf.shape(inputs)[0]], tf.shape(inputs)[1]))
        reversed_inputs = tf.reverse_sequence(inputs, lengths, seq_axis=1, batch_axis=0)
        forward = self.forward_layer(inputs, mask=mask, training=training)
        backward = self.backward_layer(reversed_inputs, mask=mask, training=training)
        if self.return_sequences:
            backward = tf.reverse_sequence(backward, lengths, seq_axis=1, batch_axis=0)
        values = tf.concat([forward, backward], axis=-1)
        if self.return_sequences and mask is not None:
            values = tf.where(mask[..., None], values, tf.zeros_like(values))
        return values

    def compute_output_shape(self, shape):
        return ((shape[0], shape[1], self.units * 2) if self.return_sequences
                else (shape[0], self.units * 2))

    def compute_mask(self, inputs, mask):
        return mask if self.return_sequences else None

    def get_config(self):
        return dict(super().get_config(), units=self.units, return_sequences=self.return_sequences)


def gathered_clone(model):
    def clone(layer):
        config = layer.get_config()
        if isinstance(layer, keras.layers.Dense):
            return PortableDense.from_config(config)
        if isinstance(layer, keras.layers.LSTM):
            return GatherLSTM.from_config(config)
        if isinstance(layer, keras.layers.Bidirectional):
            recurrent = layer.forward_layer
            if not isinstance(recurrent, keras.layers.LSTM) or layer.merge_mode != 'concat':
                raise ValueError('Portable BLSTM requires concatenated LSTM directions')
            for direction in (recurrent, layer.backward_layer):
                config = direction.get_config()
                if (config['activation'] != 'tanh' or config['recurrent_activation'] != 'sigmoid'
                        or not config['use_bias'] or config['dropout'] or config['recurrent_dropout']
                        or config['return_state'] or config['stateful']):
                    raise ValueError('Portable BLSTM supports the original stateless LSTM configuration only')
            return PortableBidirectional(recurrent.units, return_sequences=recurrent.return_sequences, name=layer.name)
        if isinstance(layer, keras.layers.TimeDistributed):
            return keras.layers.TimeDistributed(clone(layer.layer), name=layer.name)
        return layer.__class__.from_config(config)
    result = keras.models.clone_model(model, clone_function=clone)
    result.set_weights(model.get_weights())
    return result

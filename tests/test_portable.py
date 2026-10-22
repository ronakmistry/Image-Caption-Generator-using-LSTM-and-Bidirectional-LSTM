import numpy as np
import keras
import tensorflow as tf
from captioning.models import build_decoder
from captioning.portable import gathered_clone


def test_gathered_decoder_matches_reference_and_has_scalar_loss(tmp_path):
    keras.utils.set_random_seed(11)
    reference = build_decoder(4, 9, 5, units=3, embedding_dim=4)
    candidate = gathered_clone(reference)
    features = np.random.default_rng(4).normal(size=(3, 4)).astype('float32')
    tokens = np.array([[2, 0, 0, 0, 0], [2, 4, 0, 0, 0], [2, 4, 5, 6, 0]], dtype='int32')
    inputs = [features, tokens]
    np.testing.assert_allclose(reference(inputs, training=False), candidate(inputs, training=False), atol=1e-6)
    assert getattr(candidate(inputs, training=False), '_keras_mask', None) is None
    candidate.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    logs = candidate.train_on_batch(inputs, np.array([4, 5, 3]), return_dict=True)
    assert np.ndim(logs['loss']) == 0 and np.isfinite(logs['loss'])
    candidate.save(tmp_path / 'candidate.keras')
    restored = keras.models.load_model(tmp_path / 'candidate.keras')
    np.testing.assert_allclose(candidate(inputs, training=False), restored(inputs, training=False), atol=1e-6)


def test_gathered_gradient_matches_reference():
    keras.utils.set_random_seed(15)
    reference = build_decoder(4, 9, 5, units=3, embedding_dim=4)
    candidate = gathered_clone(reference)
    features = tf.ones((2, 4))
    tokens = tf.constant([[2, 0, 0, 0, 0], [2, 4, 5, 0, 0]])
    def gradients(model):
        with tf.GradientTape() as tape:
            probabilities = model([features, tokens], training=False)
            loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(tf.constant([4, 3]), probabilities))
        return [tf.convert_to_tensor(g).numpy() for g in tape.gradient(loss, model.trainable_variables)]
    for expected, actual in zip(gradients(reference), gradients(candidate)):
        np.testing.assert_allclose(expected, actual, atol=1e-6)

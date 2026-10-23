"""Original three-stage recurrent decoder, expressed with supported Keras APIs."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
from tensorflow import keras
from keras import layers, ops
from . import portable  # Register portable checkpoint layers for normal loading.


def build_encoder(name, weights=None):
    factory = {'inception': keras.applications.InceptionV3,
               'mobilenet': keras.applications.MobileNet}[name]
    model = factory(include_top=False, pooling='avg', weights=None)
    if weights is not None:
        model.load_weights(weights)
    model.trainable = False
    return model


def build_decoder(feature_dim, vocab_size, max_len, bidirectional=False, units=256, embedding_dim=300):
    image = keras.Input((feature_dim,), name='image_features')
    tokens = keras.Input((max_len,), dtype='int32', name='prefix')
    mask = ops.not_equal(tokens, 0)
    photo = layers.Dense(embedding_dim, activation='relu')(layers.Dropout(.2)(image))
    sequence = layers.Embedding(vocab_size, embedding_dim)(tokens)
    sequence = layers.Dropout(.2)(sequence)

    def recurrent(x, return_sequences):
        layer = layers.LSTM(units, return_sequences=return_sequences)
        if bidirectional:
            layer = layers.Bidirectional(layer)
        return layer(x, mask=mask)

    sequence = recurrent(sequence, True)
    sequence = layers.TimeDistributed(layers.Dense(embedding_dim))(sequence)
    sequence = layers.Add()([layers.RepeatVector(max_len)(photo), sequence])
    sequence = recurrent(layers.Dropout(.2)(sequence), True)
    sequence = recurrent(sequence, False)
    decoded = layers.Dense(units, activation='relu')(sequence)
    result = layers.Dense(vocab_size, activation='softmax')(decoded)
    model = keras.Model([image, tokens], result)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

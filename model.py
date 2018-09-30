from __future__ import absolute_import

import tensorflow as tf
import numpy as np


def padding_aware_softmax(logits, query_length, key_length, mask_future):
    """
    A numerically stable implementation of softmax that always assigns zero weight to padded values
    :param logits: A tf.Tensor of shape [B, TQ, TK]
    :param query_length: A tf.Tensor of shape [B] where each value is < TQ
    :param key_length: A tf.Tensor of shape [B] where each value is < TK
    """

    with tf.name_scope("padding_aware_softmax"):

        # Lengths to which batches are padded to
        TQ = tf.shape(logits)[1]
        TK = tf.shape(logits)[2]

        # Derive masks
        query_mask = tf.sequence_mask(lengths=query_length, maxlen=TQ, dtype=tf.int32)  # [B, TQ]
        key_mask = tf.sequence_mask(lengths=key_length, maxlen=TK, dtype=tf.int32)  # [B, TK]

        # Introduce new dimensions (we want to have a batch-wise outer product)
        query_mask = tf.expand_dims(query_mask, axis=2)     # [B, TQ, 1]
        key_mask = tf.expand_dims(key_mask, axis=1)     # [B, 1, TK]

        # Combine masks
        joint_mask = tf.cast(tf.matmul(query_mask, key_mask), tf.float32, name="joint_mask")    # [B, TQ, TK]

        # Decoder is not allowed to peek into the future to prevent cheating
        if mask_future:

            # NOTE: This is a self-attention so joint_mask actually has shape [B, TQ, TQ]
            row_ix = tf.expand_dims(tf.range(TQ), axis=1)   # [TQ, 1]
            col_ix = tf.expand_dims(tf.range(TQ), axis=0)   # [1, TQ]

            # Repeat indices
            row_ix = tf.tile(row_ix, multiples=[1, TQ])     # [TQ, TQ]
            col_ix = tf.tile(col_ix, multiples=[TQ, 1])     # [TQ, TQ]

            # Lower triangular mask
            tril_mask = tf.cast(tf.greater_equal(row_ix, col_ix), tf.float32, name="tril_mask")      # [TQ, TQ]

            # Combine with other mask
            joint_mask = tf.multiply(joint_mask, tril_mask, name="overlay_mask")     # [B, TQ, TQ]

        # Padding should not influence maximum (replace with minimum)
        logits_min = tf.reduce_min(logits, axis=2, keepdims=True, name="logits_min")      # [B, TQ, 1]
        logits_min = tf.tile(logits_min, multiples=[1, 1, TK])  # [B, TQ, TK]
        logits = tf.where(condition=joint_mask > .5,
                          x=logits,
                          y=logits_min)

        # Determine maximum
        logits_max = tf.reduce_max(logits, axis=2, keepdims=True, name="logits_max")      # [B, TQ, 1]
        logits_shifted = tf.subtract(logits, logits_max, name="logits_shifted")    # [B, TQ, TK]

        # Derive unscaled weights
        weights_unscaled = tf.exp(logits_shifted, name="weights_unscaled")

        # Apply mask
        weights_unscaled = tf.multiply(joint_mask, weights_unscaled, name="weights_unscaled_masked")     # [B, TQ, TK]

        # Derive total mass
        weights_total_mass = tf.reduce_sum(weights_unscaled, axis=2,
                                           keepdims=True, name="weights_total_mass")     # [B, TQ, 1]

        # Avoid division by zero
        weights_total_mass = tf.where(condition=tf.equal(query_mask, 1),
                                      x=weights_total_mass,
                                      y=tf.ones_like(weights_total_mass))

        # Normalize weights
        weights = tf.divide(weights_unscaled, weights_total_mass, name="normalize_attention_weights")   # [B, TQ, TK]

        return weights


def attention(query, key, value,
              query_length, key_length,
              num_heads, num_head_dims,
              mask_future=False):
    """
     Multi-head attention
     :param query: A tf.Tensor of shape [B, TQ, E]
     :param key: A tf.Tensor of shape [B, TK, E]
     :param value: A tf.Tensor of shape [B, TK, E]
     :param num_heads: How many sets of weights are desired
    """
    with tf.name_scope("attention"):

        # Make shapes compatible for tf.Dense
        B = tf.shape(query)[0]
        E = query.get_shape().as_list()[2]

        # Number of times steps (dynamic)
        TQ = tf.shape(query)[1]
        TK = tf.shape(key)[1]

        query = tf.reshape(query, shape=[B * TQ, E], name="query")   # [B * TQ, E]
        key = tf.reshape(key, shape=[B * TK, E], name="key")   # [B * TK, E]
        value = tf.reshape(value, shape=[B * TK, E], name="value")   # [B * TK, E]

        # Linear projections & single attention
        head_summaries = []
        for head_id in range(num_heads):

            # Apply three different linear projections
            head_query = tf.layers.Dense(units=num_head_dims,
                                         activation=None,
                                         use_bias=False)(query)  # [B * TQ, num_head_dims]
            head_key = tf.layers.Dense(units=num_head_dims,
                                       activation=None,
                                       use_bias=False)(key)  # [B * TK, num_head_dims]
            head_value = tf.layers.Dense(units=num_head_dims,
                                         activation=None,
                                         use_bias=False)(value)  # [B * TK, num_head_dims]
            # Reshape
            head_query = tf.reshape(head_query, shape=[B, TQ, num_head_dims])  # [B, TQ, num_head_dims]
            head_key = tf.reshape(head_key, shape=[B, TK, num_head_dims])  # [B, TK, num_head_dims]
            head_value = tf.reshape(head_value, shape=[B, TK, num_head_dims])  # [B, TK, num_head_dims]

            # Derive attention logits
            head_attention_weights = tf.matmul(head_query, tf.transpose(head_key, perm=[0, 2, 1]))  # [B, TQ, TK]

            # Rescale
            head_attention_weights = tf.divide(head_attention_weights, np.sqrt(num_head_dims))  # [B, TQ, TK]

            # Normalize weights
            head_attention_weights = padding_aware_softmax(logits=head_attention_weights,
                                                           query_length=query_length,
                                                           key_length=key_length,
                                                           mask_future=mask_future)     # [B, TQ, TK]

            # Apply weights to values
            head_summary = tf.matmul(head_attention_weights, head_value)     # [B, TQ, num_head_dims]

            # Add to collection
            head_summaries.append(head_summary)

        # Contract list of Tensors
        head_summaries = tf.stack(head_summaries, axis=2, name="head_summaries")   # [B, TQ, H, num_head_dims]

        # Project back to input embedding size (E)
        head_summaries = tf.reshape(head_summaries,
                                    shape=[B * TQ, num_heads * num_head_dims],
                                    name="head_summaries_reshaped")  # [B * TQ, H * num_head_dims]
        output = tf.layers.Dense(units=E,
                                 activation=None,
                                 use_bias=False)(head_summaries)
        output = tf.reshape(output, shape=[B, TQ, E], name="attention")   # [B, TQ, E]

        return output


def encoder_layer(x, x_length,
                  num_heads, num_head_dims,
                  num_dense_dims, dense_activation,
                  dropout_rate, is_training):
    """
    Encoder layer
    :param x: A tf.Tensor of shape [B, T, E]
    :param x_length:  A tf.Tensor of shape [B]
    :param num_heads: Number of attention heads
    :param num_head_dims: Dimension of linear projection of each head
    :param num_dense_dims: Dimension of linear projection between dense layers
    :param dense_activation: Activation function of the dense layers
    :param dropout_rate: How many neurons should be deactivated (between 0 and 1)
    :param is_training: Whether we are in training or prediction mode
    :return: A tf.Tensor of shape [B, T, E]
    """

    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    E = x.get_shape().as_list()[2]

    """ First sub-layer (self attention) """
    layer_norm_opts = {
        "begin_norm_axis": 2,
        "begin_params_axis": 2,
        "scale": False,
        "center": True
    }

    # Residual
    residual = x    # [B, T, E]

    # Layer normalization
    x = tf.contrib.layers.layer_norm(inputs=x, **layer_norm_opts)      # [B, T, E]

    # Self attention
    x = attention(query=x,
                  key=x,
                  value=x,
                  query_length=x_length,
                  key_length=x_length,
                  num_heads=num_heads,
                  num_head_dims=num_head_dims)  # [B, T, E]

    # Dropout
    x = tf.layers.Dropout(rate=dropout_rate, noise_shape=[B, 1, E])(x, training=is_training)     # [B, T, E]

    # Residual connection
    x = x + residual    # [B, T, E]

    # Layer normalization
    x = tf.contrib.layers.layer_norm(inputs=x, **layer_norm_opts)      # [B, T, E]

    """ Second sub-layer (dense) """

    # Residual
    residual = x    # [B, T, E]

    # Reshape to make output compatible with tf.layers.Dense
    x = tf.reshape(x, shape=[B * T, E])    # [B * T, E]

    # Dense
    x = tf.layers.Dense(units=num_dense_dims,
                        use_bias=True,
                        activation=dense_activation)(x)    # [B * T, num_dense_dims]
    # Dropout
    x = tf.layers.Dropout(rate=dropout_rate, noise_shape=[1, num_dense_dims])(x, training=is_training)     # [B * T, num_dense_dims]

    # Dense
    x = tf.layers.Dense(units=E,
                        use_bias=True,
                        activation=dense_activation)(x)     # [B * T, E]

    # Dropout
    x = tf.layers.Dropout(rate=dropout_rate, noise_shape=[1, E])(x, training=is_training)     # [B * T, E]

    # Reshape again
    x = tf.reshape(x, shape=[B, T, E])     # [B, T, E]

    # Residual connection
    x = x + residual     # [B, T, E]

    # Layer normalization
    x = tf.contrib.layers.layer_norm(inputs=x, **layer_norm_opts)     # [B, T, E]

    return x


def encoder(x, x_length, params, is_training):
    """ Chain of identical layers (self attention and dense layers) """

    with tf.name_scope("encoder"):

        for layer_id in range(params["num_encoder_layers"]):
            x = encoder_layer(x=x,
                              x_length=x_length,
                              num_heads=params["num_encoder_attention_heads"],
                              num_head_dims=params["num_encoder_attention_head_dims"],
                              num_dense_dims=params["num_encoder_dense_dims"],
                              dense_activation=params["encoder_dense_activation"],
                              dropout_rate=params["encoder_dropout_rate"],
                              is_training=is_training)

    return x


def decoder_layer(x, x_length,
                  encoder_out, encoder_out_length,
                  num_heads, num_head_dims, num_dense_dims,
                  dense_activation, dropout_rate, is_training):
    """
    Decoder layer
    :param x: A tf.Tensor of shape [B, T, E]
    :param x_length:  A tf.Tensor of shape [B]
    :param encoder_out: A tf.Tensor of shape [B, T', E]
    :param encoder_out_length: A tf.Tensor of shape [B]
    :param num_heads: Number of attention heads
    :param num_head_dims: Dimension of linear projection of each head
    :param num_dense_dims: Dimension of linear projection between dense layers
    :param dense_activation: Activation function of the dense layers
    :param dropout_rate: How many neurons should be deactivated (between 0 and 1)
    :param is_training: Whether we are in training or prediction mode
    :return: A tf.Tensor of shape [B, T, E]
    """

    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    E = x.get_shape().as_list()[2]

    """ First sub-layer (self attention) """
    layer_norm_opts = {
        "begin_norm_axis": 2,
        "begin_params_axis": 2,
        "scale": False,
        "center": True
    }

    # Residual
    residual = x    # [B, T, E]

    # Layer normalization
    x = tf.contrib.layers.layer_norm(inputs=x, **layer_norm_opts)      # [B, T, E]

    # Self attention
    x = attention(query=x,
                  key=x,
                  value=x,
                  query_length=x_length,
                  key_length=x_length,
                  num_heads=num_heads,
                  num_head_dims=num_head_dims,
                  mask_future=True)  # [B, T, E]

    # Dropout
    x = tf.layers.Dropout(rate=dropout_rate, noise_shape=[B, 1, E])(x, training=is_training)     # [B, T, E]

    # Residual connection
    x = x + residual    # [B, T, E]

    # Layer normalization
    x = tf.contrib.layers.layer_norm(inputs=x, **layer_norm_opts)      # [B, T, E]

    """ Second sub-layer (attention over encoder output) """

    # Residual
    residual = x    # [B, T, E]

    # Attention
    x = attention(query=x,
                  key=encoder_out,
                  value=encoder_out,
                  query_length=x_length,
                  key_length=encoder_out_length,
                  num_heads=num_heads,
                  num_head_dims=num_head_dims)  # [B, T, E]

    # Dropout
    x = tf.layers.Dropout(rate=dropout_rate, noise_shape=[B, 1, E])(x, training=is_training)  # [B, T, E]

    # Residual connection
    x = x + residual    # [B, T, E]

    # Layer normalization
    x = tf.contrib.layers.layer_norm(inputs=x, **layer_norm_opts)

    """ Third sub-layer (dense) """

    # Residual
    residual = x    # [B, T, E]

    # Reshape to make output compatible with tf.layers.Dense
    x = tf.reshape(x, shape=[B * T, E])    # [B * T, E]

    # Dense
    x = tf.layers.Dense(units=num_dense_dims,
                        use_bias=True,
                        activation=dense_activation)(x)    # [B * T, num_dense_dims]
    # Dropout
    x = tf.layers.Dropout(rate=dropout_rate, noise_shape=[1, num_dense_dims])(x, training=is_training)     # [B * T, num_dense_dims]

    # Dense
    x = tf.layers.Dense(units=E,
                        use_bias=True,
                        activation=dense_activation)(x)     # [B * T, E]

    # Dropout
    x = tf.layers.Dropout(rate=dropout_rate, noise_shape=[1, E])(x, training=is_training)     # [B * T, E]

    # Reshape again
    x = tf.reshape(x, shape=[B, T, E])     # [B, T, E]

    # Residual connection
    x = x + residual     # [B, T, E]

    # Layer normalization
    x = tf.contrib.layers.layer_norm(inputs=x, **layer_norm_opts)     # [B, T, E]

    return x


def decoder(prev, prev_length, encoder_out, encoder_out_length, num_words, params, is_training):
    """ Chain of identical layers (self attention, attention over encoder and dense layers) """
    x = prev

    with tf.name_scope("decoder"):
        for layer_id in range(params["num_decoder_layers"]):
            x = decoder_layer(x=x,
                              x_length=prev_length,
                              encoder_out=encoder_out,
                              encoder_out_length=encoder_out_length,
                              num_heads=params["num_encoder_attention_heads"],
                              num_head_dims=params["num_encoder_attention_head_dims"],
                              num_dense_dims=params["num_encoder_dense_dims"],
                              dense_activation=params["encoder_dense_activation"],
                              dropout_rate=params["encoder_dropout_rate"],
                              is_training=is_training)

        # Probability estimate for each token
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        E = params["embedding_size"]
        x = tf.reshape(x, shape=[B * T, E])
        x = tf.layers.Dense(units=num_words + 1,    # There is one OOV token
                            activation=None,
                            use_bias=True)(x)   # [B * T, num_words]
        x = tf.reshape(x, shape=[B, T, num_words + 1])      # [B, T, num_words + 1]

    return x


def preprocess_text(lang, text, vocabulary_file, num_words, embedding_size, is_pre_trained):

    # Map words to integer IDs
    table = tf.contrib.lookup.index_table_from_file(vocabulary_file=vocabulary_file,
                                                    num_oov_buckets=1)

    text_token_ids = table.lookup(text)

    # Embedding
    W = tf.get_variable(
        name="W_{}_embedding".format(lang),
        shape=[num_words + 1, embedding_size],
        initializer=None if is_pre_trained else tf.initializers.random_normal,
        trainable=not is_pre_trained,   # No fine-tuning of pre-trained embeddings
        dtype=tf.float32)

    embedding = tf.nn.embedding_lookup(params=W, ids=text_token_ids)    # [B, T, E]

    # Positional encoding (uses same notation as in original paper)
    T = tf.shape(embedding)[1]
    d_model = embedding_size
    pos = tf.cast(tf.tile(tf.expand_dims(tf.range(T), axis=0), multiples=[d_model, 1]), tf.float32)  # [E, T]
    i = tf.cast(tf.tile(tf.expand_dims(tf.range(d_model), axis=1), multiples=[1, T]), tf.float32)    # [E, T]

    # Sine waves
    sine = tf.sin(tf.divide(pos, tf.pow(float(10**4), tf.divide(i, d_model))))     # [E, T]
    cosine = tf.cos(tf.divide(pos, tf.pow(float(10**4), tf.divide(i, d_model))))   # [E, T]
    cosine = tf.manip.roll(cosine, shift=1, axis=0)

    # Alternate between waves depending on parity
    even_mask = tf.equal(tf.mod(tf.range(d_model), 2), 0)   # [E]
    joint_pos = tf.where(condition=even_mask, x=sine, y=cosine)     # [E, T]
    joint_pos = tf.transpose(joint_pos)     # [T, E]

    # Magnitude of positional embedding
    gamma = tf.get_variable(name="gamma_{}".format(lang),
                            shape=[],
                            initializer=tf.initializers.ones,
                            trainable=True,
                            dtype=tf.float32)

    # Apply positional encoding
    embedding = tf.add(embedding, gamma * joint_pos, name="composed_embedding")   # [B, T, E]

    return text_token_ids, embedding, W


def translate_indices(indices, vocab_fname, vocab_num_words, mask=None):
    if mask is not None:
        indices = tf.where(condition=mask > .5,
                           x=indices,
                           y=tf.ones_like(indices))  # [B, T - 1]

    inverse_map = tf.contrib.lookup.index_to_string_table_from_file(vocabulary_file=vocab_fname,
                                                                    vocab_size=vocab_num_words,
                                                                    default_value="<UNK>")
    indices = inverse_map.lookup(indices)  # [B, T - 1]
    return tf.reduce_join(indices, axis=1, separator=" ")


def model_fn(features, labels, mode, params):

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    is_pre_trained = ("pre_trained_embedding_de" in params) and ("pre_trained_embedding_en" in params)

    # Lengths
    de_text_length = features["de_text_length"]
    en_text_length = features["en_text_length"]

    # Text preprocessing
    de_token_ids, de_text, W_de = preprocess_text(lang="de",
                                                  text=features["de_text"],
                                                  vocabulary_file=params["de_vocab_fname"],
                                                  num_words=params["de_vocab_num_words"],
                                                  embedding_size=params["embedding_size"],
                                                  is_pre_trained=is_pre_trained)
    en_token_ids, en_text, W_en = preprocess_text(lang="en",
                                                  text=features["en_text"],
                                                  vocabulary_file=params["en_vocab_fname"],
                                                  num_words=params["en_vocab_num_words"],
                                                  embedding_size=params["embedding_size"],
                                                  is_pre_trained=is_pre_trained)

    # Encoder
    encoder_out = encoder(x=de_text,
                          x_length=de_text_length,
                          params=params,
                          is_training=is_training)

    # Decoder
    decoder_out = decoder(prev=en_text,
                          prev_length=en_text_length,
                          encoder_out=encoder_out,
                          encoder_out_length=de_text_length,
                          num_words=params["en_vocab_num_words"],
                          params=params,
                          is_training=is_training)

    if mode == tf.estimator.ModeKeys.PREDICT:

        # Only last time step is relevant
        decoder_out = decoder_out[:, -1, :]     # [B, num_words]

        # Decode probabilities
        decoder_out = tf.nn.softmax(decoder_out, axis=1)    # [B, num_words]

        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=decoder_out)

    # Shift decoder output (first token is always start token)
    decoder_out = decoder_out[:, :-1, :]    # [B, T - 1, num_words]
    reference = en_token_ids[:, 1:]     # [B, T - 1]

    # Cross entropy
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_out, labels=reference)     # [B, T - 1]

    # Loss incurred from padding should be masked out
    T = tf.shape(en_text)[1]
    mask = tf.sequence_mask(lengths=en_text_length - 1,
                            maxlen=T - 1,
                            dtype=tf.float32)   # [B, T - 1]

    # Apply mask
    loss = tf.multiply(mask, loss)  # [B, T - 1]

    # Calculate mean (ignoring padding)
    mask_mass = tf.reduce_sum(mask, axis=1)     # [B]
    loss = tf.reduce_sum(loss, axis=1)   # [B]
    loss = tf.divide(loss, mask_mass)
    loss = tf.reduce_mean(loss)     # []

    # Decay learning rate
    learning_rate = tf.train.cosine_decay_restarts(learning_rate=params["learning_rate"],
                                                   global_step=tf.train.get_global_step(),
                                                   first_decay_steps=params["learning_first_decay_steps"],
                                                   t_mul=params["learning_t_mul"],
                                                   m_mul=params["learning_m_mul"],
                                                   alpha=params["learning_alpha"])

    # Optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=params["momentum"],
                                           use_nesterov=True)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Input sequence/translation as strings (for debugging purposes)
    input_sentence = tf.reduce_join(features["de_text"], separator=" ", axis=1)
    translation = tf.argmax(decoder_out, axis=2)    # [B, T - 1]
    translation = translate_indices(indices=translation,
                                    vocab_fname=params["en_vocab_fname"],
                                    vocab_num_words=params["en_vocab_num_words"],
                                    mask=mask)
    ground_truth_text = translate_indices(indices=en_token_ids[:, :-1],
                                          vocab_fname=params["en_vocab_fname"],
                                          vocab_num_words=params["en_vocab_num_words"],
                                          mask=mask)

    # TensorBoard
    tf.summary.scalar(name="ground_truth_prob", tensor=tf.exp(-loss))
    tf.summary.text(name="input_sentence", tensor=input_sentence)
    tf.summary.text(name="translation", tensor=translation)
    tf.summary.text(name="ground_truth_text", tensor=ground_truth_text)
    tf.summary.scalar(name="decayed_learning_rate", tensor=learning_rate)

    # Pre-trained embeddings
    scaffold = None
    if is_pre_trained:
        # Accounts for one extra OOV token
        pad_width = [[0, 1], [0, 0]]
        W_init_de = np.pad(np.load(params["pre_trained_embedding_de"]), pad_width=pad_width, mode="mean")
        W_init_en = np.pad(np.load(params["pre_trained_embedding_en"]), pad_width=pad_width, mode="mean")

        def _init_fn(_, sess):
            sess.run([W_de.initializer, W_en.initializer],
                     feed_dict={W_de.initial_value: W_init_de,
                                W_en.initial_value: W_init_en})
        scaffold = tf.train.Scaffold(init_fn=_init_fn)

    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                      loss=loss,
                                      train_op=train_op,
                                      scaffold=scaffold)

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from argparse import ArgumentParser
import os
import yaml
import random
import string
import tempfile
from functools import partial
import itertools
import numpy as np
import spacy
import tensorflow as tf
import textwrap

from tensorflow.python import debug as tf_debug
from model import model_fn
from translation import Translator

# spaCy objects
nlp_de = spacy.load("de")

# TF logging
tf.logging.set_verbosity(tf.logging.INFO)


def _parse(record):

    # Define data types
    features = {
        "en_text": tf.VarLenFeature(dtype=tf.string),
        "de_text": tf.VarLenFeature(dtype=tf.string)
    }

    return tf.parse_single_example(record, features=features)


def _densify(record, field):
    record[field] = tf.sparse_tensor_to_dense(record[field], default_value="")
    return record


def _add_length(record, field):
    record["{}_length".format(field)] = tf.shape(record[field])[0]
    return record


def _extract_length(record, field):
    return record["{}_length".format(field)]


def _prepend_start_token(record, field):
    record[field] = tf.pad(record[field],
                           paddings=[[1, 0]],
                           constant_values="<<START>>")
    return record


def _append_end_token(record, field):
    record[field] = tf.pad(record[field],
                           paddings=[[0, 1]],
                           constant_values="<<END>>")
    return record


def get_input_fn(fname, config):

    # Create dataset
    data = tf.data.TFRecordDataset(filenames=[fname])

    # Parse single records
    data = data.map(_parse)

    # Densify
    data = data.map(partial(_densify, field="en_text"))
    data = data.map(partial(_densify, field="de_text"))

    # Prepend start tokens
    data = data.map(partial(_prepend_start_token, field="en_text"))
    data = data.map(partial(_prepend_start_token, field="de_text"))

    # Append end tokens
    data = data.map(partial(_append_end_token, field="en_text"))
    data = data.map(partial(_append_end_token, field="de_text"))

    # Add lengths of texts
    data = data.map(partial(_add_length, field="en_text"))
    data = data.map(partial(_add_length, field="de_text"))

    # Epochs (doing this before the following steps is computationally more costly but improves variability of batches)
    data = data.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=config["shuffle_buffer_size"],
                                                         count=config["num_epochs"]))

    # Bucket by sequence length)
    bucket_boundaries = config["bucket_boundaries"]
    bucket_batch_sizes = [config["batch_size"]] * (len(bucket_boundaries) + 1)
    data = data.apply(tf.contrib.data.bucket_by_sequence_length(element_length_func=partial(_extract_length, field="de_text"),
                                                                bucket_boundaries=bucket_boundaries,
                                                                bucket_batch_sizes=bucket_batch_sizes,
                                                                padded_shapes={
                                                                    "en_text": tf.TensorShape([None]),
                                                                    "de_text": tf.TensorShape([None]),
                                                                    "en_text_length": tf.TensorShape([]),
                                                                    "de_text_length": tf.TensorShape([])
                                                                }))

    return data


def verify_data(input_fn,
                de_vocab_fname, de_vocab_num_words,
                en_vocab_fname, en_vocab_num_words,
                num_chars=40):
    """ Prints one batch of the input data """

    # Auxiliary functions
    def _format_line(*args):
        return "\t".join(["{}"] * len(args)).format(*map(lambda arg: arg.center(num_chars), args))

    # Get iterator
    it = input_fn().make_one_shot_iterator()
    next_el = it.get_next()

    # Index mappings
    table_de = tf.contrib.lookup.index_table_from_file(vocabulary_file=de_vocab_fname, num_oov_buckets=1)
    table_en = tf.contrib.lookup.index_table_from_file(vocabulary_file=en_vocab_fname, num_oov_buckets=1)

    # Apply forward mappings
    de_text_ids = table_de.lookup(next_el["de_text"])
    en_text_ids = table_en.lookup(next_el["en_text"])

    # Inverse vocab
    table_de_inv = tf.contrib.lookup.index_to_string_table_from_file(vocabulary_file=de_vocab_fname,
                                                                     vocab_size=de_vocab_num_words,
                                                                     default_value="<UNK>")

    table_en_inv = tf.contrib.lookup.index_to_string_table_from_file(vocabulary_file=en_vocab_fname,
                                                                     vocab_size=en_vocab_num_words,
                                                                     default_value="<UNK>")

    # Apply backward mappings
    de_text_mapped = table_de_inv.lookup(de_text_ids)
    en_text_mapped = table_en_inv.lookup(en_text_ids)

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        output = sess.run({
            "de_text": next_el["de_text"],
            "en_text": next_el["en_text"],
            "de_text_ids": de_text_ids,
            "en_text_ids": en_text_ids,
            "de_text_mapped": de_text_mapped,
            "en_text_mapped": en_text_mapped,
            "de_text_length": next_el["de_text_length"],
            "en_text_length": next_el["en_text_length"]
        })

        # Unfold
        en_text, en_text_length = output["en_text"].tolist(), output["en_text_length"].tolist()
        de_text, de_text_length = output["de_text"].tolist(), output["de_text_length"].tolist()
        en_text_mapped = output["en_text_mapped"].tolist()
        de_text_mapped = output["de_text_mapped"].tolist()

        for example_id in range(len(en_text)):
            en_example = textwrap.wrap(" ".join(map(lambda _: _.decode("UTF-8"), en_text[example_id])), width=num_chars)
            de_example = textwrap.wrap(" ".join(map(lambda _: _.decode("UTF-8"), de_text[example_id])), width=num_chars)
            en_example_mapped = textwrap.wrap(" ".join(map(lambda _: _.decode("UTF-8"), en_text_mapped[example_id])), width=num_chars)
            de_example_mapped = textwrap.wrap(" ".join(map(lambda _: _.decode("UTF-8"), de_text_mapped[example_id])), width=num_chars)

            # Paragraphs should be next to each other
            aligned = itertools.starmap(_format_line, itertools.zip_longest(en_example, en_example_mapped,
                                                                            de_example, de_example_mapped, fillvalue=""))
            print("[{}]".format(example_id))
            print(os.linesep.join(aligned) + os.linesep)


def create_interactive_data(user_input):

    def _token_filter(token):
        return token.is_alpha or token.text in (".", ",")

    # Derive tokens
    user_input_tokens = ["<<START>>"] + [token.text.lower() for token in nlp_de(user_input) if _token_filter(token)] + ["<<END>>"]
    user_input_tokens = np.array([token.encode("UTF-8") for token in user_input_tokens], dtype=np.object)

    # Assemble data set
    data = {"de_text": user_input_tokens,
            "de_text_length": user_input_tokens.size}

    return data


def resolve_symbols(params):
    """ Map strings to TensorFlow functions """
    resolved = dict(params)
    for k, v in resolved.items():
        if isinstance(v, str) and v.startswith("tf."):
            resolved[k] = eval(v)
    return resolved


# Necessary because this is only available in python >= 3.6
def choices(population, k):
    return [random.choice(population) for _ in range(k)]


def main():
    # Instantiate parser
    parser = ArgumentParser()

    # Add arguments
    parser.add_argument("--mode", "-m",
                        help="Whether to 'train' or to 'predict'",
                        required=True)

    parser.add_argument("--prediction_scheme",
                        help="Whether to use a beam search ('beam'), a greedy search ('greedy')"
                             " or a one-time lookup ('one_time')")

    parser.add_argument("--data", "-d",
                        help="The path to the data in TFRecords format")

    parser.add_argument("--de_vocab",
                        help="The GERMAN vocabulary list",
                        required=True)

    parser.add_argument("--en_vocab",
                        help="The ENGLISH vocabulary list",
                        required=True)

    parser.add_argument("--pre_trained_embedding_de",
                        help="The pre-trained GERMAN embeddings (.npy)")

    parser.add_argument("--pre_trained_embedding_en",
                        help="The pre-trained ENGLISH embeddings (.npy)")

    parser.add_argument("--config", "-c",
                        help="The YAML file that defines the model/training",
                        required=True)

    parser.add_argument("--temp", "-t",
                        help="Where to store temporary files by TensorFlow",
                        default=tempfile.gettempdir())

    parser.add_argument("--verify", "-v",
                        help="Whether to print the input data for verification purposes",
                        default=False,
                        action="store_true")

    parser.add_argument("--model_dir",
                        help="Where the checkpoint of a trained TF model resides",
                        required=False)

    parser.add_argument("--debug",
                        help="Whether to activate TensorFlow's debug mode",
                        default=False,
                        action="store_true")

    parser.add_argument("--tensorboard",
                        help="Whether to spawn a TensorBoard daemon",
                        default=False,
                        action="store_true")

    # Parse
    args = parser.parse_args()

    # Input assertions
    assert args.mode in {"train", "predict"}
    assert args.mode == "train" or args.prediction_scheme in ("beam", "greedy", "one_time")
    assert os.path.exists(args.config), "Config does not exist!"

    # Parse config
    config = yaml.load(open(args.config, "r"))
    assert {"batch_size", "num_epochs", "shuffle_buffer_size"} <= set(config.keys())

    # Resolve symbols
    config = resolve_symbols(config)

    # Add vocabulary files
    if args.pre_trained_embedding_de and args.pre_trained_embedding_en:
        config["pre_trained_embedding_de"] = args.pre_trained_embedding_de
        config["pre_trained_embedding_en"] = args.pre_trained_embedding_en

    config["en_vocab_fname"] = args.en_vocab
    config["de_vocab_fname"] = args.de_vocab
    config["en_vocab_num_words"] = sum(1 for _ in open(config["en_vocab_fname"], "r"))
    config["de_vocab_num_words"] = sum(1 for _ in open(config["de_vocab_fname"], "r"))

    # Random sub-directory
    if args.mode == "train":
        assert os.path.exists(args.data), "Data does not exist!"
        if args.model_dir:
            # Improve existing model
            model_dir = args.model_dir
        else:
            model_dir = os.path.join(args.temp, "TF_" + "".join(choices(string.ascii_lowercase, k=7)))
            print("Created TF temporary directory: {}".format(model_dir))
    else:
        assert args.model_dir is not None, "You have to specify a valid model directory in prediction mode!"
        model_dir = args.model_dir

    # Get input function
    input_fn = partial(get_input_fn, fname=args.data, config=config)

    if args.verify:
        verify_data(input_fn,
                    de_vocab_fname=config["de_vocab_fname"],
                    de_vocab_num_words=config["de_vocab_num_words"],
                    en_vocab_fname=config["en_vocab_fname"],
                    en_vocab_num_words=config["en_vocab_num_words"])
    else:
        # Create estimator
        session_config = tf.ConfigProto(log_device_placement=config["log_device_placement"])
        estimator_config = tf.estimator.RunConfig(session_config=session_config,
                                                  save_checkpoints_secs=60 * config["save_checkpoints_mins"],
                                                  keep_checkpoint_max=config["keep_checkpoint_max"])
        estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir, params=config, config=estimator_config)

        if args.mode == "train":

            # Train estimator
            hooks = [tf_debug.LocalCLIDebugHook()] if args.debug else None
            estimator.train(input_fn, hooks=hooks, steps=None)
        else:

            # Concurrent translator (avoids having to reload TF checkpoint after each generated token)
            translator = Translator(estimator)

            # Build inverse vocabulary (from integer IDs to tokens)
            inv_vocab = [token.strip() for token in open(config["en_vocab_fname"], "r")] + ["<UNK>"]

            while True:

                # Obtain user input
                print("Type some German sentence and hit ENTER (CTRL-C to quit)")
                german_input = input()

                # Create dataset containing this input
                start_dict = create_interactive_data(user_input=german_input)

                # Get translation
                if args.prediction_scheme == "beam":
                    translator.beam_predict(start_dict=start_dict, inv_vocab=inv_vocab, beam_size=config["beam_size"])
                elif args.prediction_scheme == "greedy":
                    translator.greedy_predict(start_dict=start_dict, inv_vocab=inv_vocab)
                else:
                    start_dict.update({
                        "en_text": np.array(["<<START>>"], dtype=np.object),
                        "en_text_length": 1
                    })
                    prediction = translator.predict(past=start_dict)
                    ix = np.argsort(-prediction)[:20]
                    tokens = [inv_vocab[i] for i in ix]
                    print(tokens)


if __name__ == "__main__":
    main()
import tensorflow as tf
import numpy as np
from queue import Queue
from threading import Thread

START_TOKEN_ID = 0
END_TOKEN_ID = 1

# Formatting
BOLD = '\033[1m'
FORMAT_END = '\033[0m'


class Translator:
    """
    Idea: https://medium.com/element-ai-research-lab/multithreaded-predictions-with-tensorflow-estimators-eb041861da07
    """
    def __init__(self, estimator):
        self.estimator = estimator

        # Queues
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)

        # Thread
        self.thread = Thread(target=self.enqueue_output_queue, daemon=True)
        self.thread.start()

    def dequeue_from_input_queue(self):
        while True:
            yield self.input_queue.get()

    def enqueue_output_queue(self):
        for prediction in self.estimator.predict(input_fn=self.queued_input_fn):
            self.output_queue.put(prediction)

    def greedy_predict(self, start_dict, inv_vocab, maxsize=100):
        inv_vocab = np.array(inv_vocab)

        # Updated iteratively
        generated_token_ids = np.zeros(shape=[maxsize + 1], dtype=np.int32)
        generated_token_probs = np.zeros(shape=[maxsize + 1], dtype=np.float32)
        generated_tokens = np.empty(shape=[maxsize + 1], dtype=np.object)
        generated_tokens[0] = "<<START>>"

        # Add dummy data
        iterative_dict = dict(start_dict)
        iterative_dict["en_text_length"] = 0

        i = 1
        while i < maxsize + 1:

            # Update input dictionary
            iterative_dict["en_text_length"] += 1
            iterative_dict["en_text"] = generated_tokens[:i]

            # Generate next token
            next_token_probs = self.predict(past=iterative_dict)
            next_token_id = np.argmax(next_token_probs)

            if next_token_id == END_TOKEN_ID:
                break

            # Save
            generated_token_ids[i] = next_token_id
            generated_token_probs[i] = next_token_probs[next_token_id]
            generated_tokens[i] = inv_vocab[next_token_id]

            i = i + 1

        # Trim translation
        generated_tokens = generated_tokens[:i].tolist()
        print(generated_tokens)
        generated_token_probs = generated_token_probs[:i]

        # Print
        print(" ".join(generated_tokens))
        print(np.array2string(generated_token_probs, precision=3))

    def beam_predict(self, start_dict, inv_vocab, beam_size, maxsize=50):

        # Map integer IDs to tokens
        token_map = np.vectorize(inv_vocab.__getitem__, otypes=[np.object])

        # Updated iteratively
        beam_token_ids = np.zeros(shape=[beam_size, maxsize + 1], dtype=np.int32)
        beam_token_probs = np.zeros(shape=[beam_size, maxsize + 1], dtype=np.float32)
        beam_tokens = np.empty(shape=[beam_size, maxsize + 1], dtype=np.object)
        beam_token_probs[:, 0] = 1.
        beam_tokens[:, 0] = b"<<START>>"

        # Add dummy data
        iterative_dict = dict(start_dict)
        iterative_dict["en_text_length"] = 0

        vocab_size = len(inv_vocab)
        i = 1
        while i < maxsize + 1:

            # Update input dictionary
            iterative_dict["en_text_length"] += 1

            # Accounts of out-of-vocab bucket
            beam_order = np.zeros(shape=[beam_size, vocab_size], dtype=np.int32)     # [B, B]
            beam_probs = np.zeros(shape=[beam_size, vocab_size], dtype=np.float32)   # [B, B]

            for beam_id in range(beam_size):

                # Assign text already generated
                iterative_dict["en_text"] = beam_tokens[beam_id, :i]

                # Generate next token
                next_token_probs = self.predict(past=iterative_dict)

                # Order by probability
                next_token_order = np.argsort(-next_token_probs)

                # Update
                beam_order[beam_id, :] = next_token_order
                beam_probs[beam_id, :] = next_token_probs

            if i == 1:
                order = np.argsort(-beam_probs[0, :])[:beam_size]
                order = np.stack([np.arange(beam_size), order], axis=1)
            else:
                # Select most promising chains
                current_total_prob = np.sum(np.log(beam_token_probs[:, :i]), axis=1, keepdims=True)     # [B, 1]
                continuation = np.add(current_total_prob, np.log(beam_probs))/i  # [B, B]
                order = np.dstack(np.unravel_index(np.argsort(-continuation.ravel()), continuation.shape))
                order = order[0, :beam_size, :]

            # Update chains
            _beam_token_probs_tmp = np.zeros_like(beam_token_probs)
            _beam_token_ids_tmp = np.zeros_like(beam_token_ids)
            for k, row in enumerate(order):
                beam_id, proposal_id = row.tolist()

                # Copy old data
                _beam_token_probs_tmp[k, :i] = beam_token_probs[beam_id, :i]
                _beam_token_ids_tmp[k, :i] = beam_token_ids[beam_id, :i]

                # Add new data
                _beam_token_probs_tmp[k, i] = beam_probs[beam_id, proposal_id]
                _beam_token_ids_tmp[k, i] = proposal_id

            beam_token_ids[:, :i+1] = _beam_token_ids_tmp[:, :i+1]
            beam_token_probs[:, :i+1] = _beam_token_probs_tmp[:, :i+1]
            beam_tokens[:, :i+1] = token_map(_beam_token_ids_tmp[:, :i+1])

            i = i + 1

        # Trim until end token
        end_ix = np.argmax(beam_token_ids == END_TOKEN_ID, axis=1)
        end_ix = np.where(end_ix == 0, maxsize, end_ix)
        end_mask = np.less(np.tile(np.expand_dims(np.arange(maxsize + 1), axis=0), reps=[beam_size, 1]),
                           np.expand_dims(end_ix, axis=1))
        beam_token_probs[~end_mask] = 1.

        # Select chain w. highest probability
        scores = np.divide(np.sum(np.log(beam_token_probs), axis=1), end_ix)
        scores_order = np.argsort(-scores)
        for ix in np.nditer(scores_order):
            tokens_ix = beam_token_ids[ix, 1:end_ix[ix]]
            tokens = token_map(tokens_ix)
            tokens_prob = beam_token_probs[ix, 1:end_ix[ix]]

            # Decode UTF-8
            generated_tokens = " ".join(tokens)
            print(generated_tokens)
            print(np.array2string(tokens_prob, precision=3).replace("\n", ""))
            print("{}Score: {:.3f}{}".format(BOLD, scores[ix], FORMAT_END))
            print()

    def predict(self, past):

        # Enqueue elements that were already generated
        self.input_queue.put(past)

        # Obtain prediction from output queue
        prediction = self.output_queue.get()

        return prediction

    def queued_input_fn(self):

        # Data types
        output_types = {
            "de_text": tf.string,
            "de_text_length": tf.int32,
            "en_text": tf.string,
            "en_text_length": tf.int32
        }

        # Data shapes (variable number of words within sentence)
        output_shapes = {
            "de_text": tf.TensorShape([None]),
            "de_text_length": tf.TensorShape([]),
            "en_text": tf.TensorShape([None]),
            "en_text_length": tf.TensorShape([])

        }

        # Define data set
        data = tf.data.Dataset.from_generator(generator=self.dequeue_from_input_queue,
                                              output_types=output_types,
                                              output_shapes=output_shapes)

        # For the sake of compatibility
        data = data.batch(1)

        return data

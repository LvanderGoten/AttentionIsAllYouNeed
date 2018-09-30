from argparse import ArgumentParser
import tensorflow as tf
from tqdm import tqdm
import spacy


# Tokenizer
disable = ["ner", "tagger", "parser", "textcat"]
nlp_en = spacy.load("en", disable=disable)
nlp_de = spacy.load("de", disable=disable)


def _token_filter(token):
    return token.is_alpha or token.text in (".", ",")


def line_generator(de_fname, en_fname):
    num_examples = sum(1 for _ in open(de_fname, "r"))
    with open(de_fname, "r") as f_de, open(en_fname, "r") as f_en:
        for de_doc, en_doc in tqdm(zip(nlp_de.pipe(f_de),
                                       nlp_en.pipe(f_en)), total=num_examples):
            yield de_doc, en_doc


def parse_doc(de_doc, en_doc):
    de_tokens = [token.text.lower() for token in de_doc if _token_filter(token)]
    en_tokens = [token.text.lower() for token in en_doc if _token_filter(token)]
    return de_tokens, en_tokens


def main():
    # Instantiate parser
    parser = ArgumentParser()

    parser.add_argument("--de",
                        help="German text file",
                        required=True)

    parser.add_argument("--en",
                        help="English text file",
                        required=True)

    parser.add_argument("--output",
                        help="Output TFRecords",
                        required=True)

    parser.add_argument("--output_de_vocab",
                        help="German vocabulary",
                        required=True)

    parser.add_argument("--output_en_vocab",
                        help="English vocabulary",
                        required=True)
    # Parse
    args = parser.parse_args()

    # Auxiliary functions
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    # Open file
    writer = tf.python_io.TFRecordWriter(args.output)

    # Vocabulary
    de_vocab = set()
    en_vocab = set()
    for de_doc, en_doc in line_generator(args.de, args.en):

        # Filter
        de_tokens, en_tokens = parse_doc(de_doc, en_doc)

        # Update vocabulary
        de_vocab.update(de_tokens)
        en_vocab.update(en_tokens)

        # Define feature
        feature = {
            "en_text": _bytes_feature(map(lambda s: s.encode("UTF-8"), en_tokens)),
            "de_text": _bytes_feature(map(lambda s: s.encode("UTF-8"), de_tokens))
        }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write out
        writer.write(example.SerializeToString())

    # Close file
    writer.close()

    de_vocab = ["<<START>>", "<<END>>"] + list(de_vocab)
    en_vocab = ["<<START>>", "<<END>>"] + list(en_vocab)

    # Write vocabulary
    with open(args.output_de_vocab, encoding="utf-8", mode="w") as f_de, open(args.output_en_vocab, encoding="utf-8", mode="w") as f_en:
        f_de.write("\n".join(de_vocab))
        f_en.write("\n".join(en_vocab))


if __name__ == "__main__":
    main()
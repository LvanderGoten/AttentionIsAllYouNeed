from __future__ import print_function

import os
import pathlib
from argparse import ArgumentParser
from bs4 import BeautifulSoup, NavigableString
import itertools
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import spacy

# spaCy objects
nlp_en = spacy.load("en")
nlp_de = spacy.load("de")


class ParallelBook:
    def __init__(self, en_book_name, de_book_name):
        self.en_book_name = en_book_name
        self.de_book_name = de_book_name
        self.chapters = []

    def add_parallel_chapter(self, chapter_id, en_content, de_content):

        if len(en_content) == len(de_content):
            self.chapters.append(ParallelChapter(chapter_id, *ParallelBook.postprocess_chapter(en_content, de_content)))

    @staticmethod
    def postprocess_chapter(en_content, de_content):
        """ Eliminate ill-defined verse and lower case them (as the size of the dataset is quite limited) """

        def _token_filter(token):
            return token.is_alpha or token.text in (".", ",")

        _en_content, _de_content = [], []
        for en_verse, de_verse in zip(en_content, de_content):
            if en_verse and de_verse:

                # Tokenize
                _en_content.append([token.text for token in nlp_en(en_verse.lower()) if _token_filter(token)])
                _de_content.append([token.text for token in nlp_de(de_verse.lower()) if _token_filter(token)])

        return _en_content, _de_content


class ParallelChapter:
    def __init__(self, chapter_id, en_content, de_content):
        self.chapter_id = os.path.splitext(chapter_id)[0]

        self.en_content = en_content
        self.de_content = de_content


def parse_htm(fname):
    """
    Parses a Bible HTML file
    :param fname: A path to a HTML file
    :return: A list of parsed Bible verses
    """

    def _construct_string(span):
        """ Accounts for the fact that verses are not properly scoped """
        _verses = []

        if span:
            curr_el, curr_verse = span, ""
            while curr_el.next_sibling:

                # Advance
                curr_el = curr_el.next_sibling

                if hasattr(curr_el, "attrs") and curr_el.attrs.get("class") == ["verse"]:
                    # Terminate string and append to verses
                    _verses.append(curr_verse.strip())
                    curr_verse = ""
                else:
                    if isinstance(curr_el, NavigableString):
                        curr_verse += str(curr_el).replace(os.linesep, " ")
                    elif hasattr(curr_el, "text"):
                        curr_verse += curr_el.text

        return _verses

    book_name, verses = "", []
    try:
        with open(fname, "r") as htm:

            # Instantiate parser
            soup = BeautifulSoup(htm, "html.parser")

            # Book name
            book_name = soup.find("div", {"class": "textHeader"}).find("h2").text

            # Find div container
            text_body = soup.find("div", {"id": "textBody"})

            # Get children
            children = text_body.find("p")

            if children:
                start_span = children.find("span", {"class": "verse"})
                verses = _construct_string(start_span)

    except UnicodeDecodeError:
        # There is one file that has a corrupted encoding
        print("Corrupted encoding detected")

    return book_name, verses


def print_corpora(books, num_books=300, num_chars=70, delimiter="="):
    """ Prints the parallel corpora where verses are separated by a delimiter """

    for _, book in itertools.islice(books.items(), num_books):
        en_book_name, de_book_name = book.en_book_name, book.de_book_name

        print("{}/{}".format(en_book_name.upper(), de_book_name.upper()))
        for chapter in book.chapters:
            en_content, de_content = chapter.en_content, chapter.de_content

            print("\tCHAPTER {}".format(chapter.chapter_id))
            for i, (en_verse, de_verse) in enumerate(itertools.zip_longest(en_content, de_content, fillvalue="")):
                print("\t\t{}: {} {} {}".format(i, " ".join(en_verse)[:num_chars].ljust(num_chars),
                                                delimiter,
                                                " ".join(de_verse)[:num_chars].ljust(num_chars)))
            print(os.linesep)

        print(os.linesep)


def print_statistics(books):
    chapters = [chapter for book in books.values() for chapter in book.chapters]
    chapter_lengths = pd.Series([len(chapter.en_content) for chapter in chapters], name="chapter_lengths")
    print("Chapter Lengths:")
    print(chapter_lengths.describe())
    print("Number of verses: {}".format(sum(chapter_lengths)))


def save_verses(books, output_en_fname, output_de_fname, output_en_vocab_fname, output_de_vocab_fname):
    chapters = [chapter for book in books.values() for chapter in book.chapters]

    de_vocab, en_vocab = set(), set()
    with open(output_en_fname, "w") as f_english, open(output_de_fname, "w") as f_german:
        for chapter in chapters:
            for en_verse, de_verse in zip(chapter.en_content, chapter.de_content):
                f_english.write("{}\n".format(" ".join(en_verse)))
                f_german.write("{}\n".format(" ".join(de_verse)))

                # Add to vocabulary
                de_vocab.update(de_verse)
                en_vocab.update(en_verse)

    # Save vocabulary
    en_vocab = ["<<START>>", "<<END>>"] + sorted(en_vocab)
    de_vocab = ["<<START>>", "<<END>>"] + sorted(de_vocab)
    with open(output_en_vocab_fname, "w") as f_en_vocab, open(output_de_vocab_fname, "w") as f_de_vocab:
        f_en_vocab.writelines("\n".join(en_vocab))
        f_de_vocab.writelines("\n".join(de_vocab))


def save_verses_as_tf_records(books, output_fname):
    chapters = [chapter for book in books.values() for chapter in book.chapters]

    # Auxiliary functions
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    # Open file
    writer = tf.python_io.TFRecordWriter(output_fname)

    for chapter in chapters:
        for en_verse, de_verse in zip(chapter.en_content, chapter.de_content):

            # Define feature
            feature = {
                "en_text": _bytes_feature(map(lambda s: s.encode("UTF-8"), en_verse)),
                "de_text": _bytes_feature(map(lambda s: s.encode("UTF-8"), de_verse))
            }

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write out
            writer.write(example.SerializeToString())

    # Close file
    writer.close()


def main():
    # Instantiate parser
    parser = ArgumentParser()

    # Add arguments
    parser.add_argument("--bible_dir", help="Where the parallel Bible resides", required=True)
    parser.add_argument("--output_en", help="Where the English verses should be stored", required=True)
    parser.add_argument("--output_de", help="Where the German verses should be stored", required=True)
    parser.add_argument("--output_en_vocab", help="Where the vocab. of English verses should be stored", required=True)
    parser.add_argument("--output_de_vocab", help="Where the vocab. German verses should be stored", required=True)
    parser.add_argument("--output_tf", help="Where the TFRecords should be stored", required=True)

    # Parse
    args = parser.parse_args()

    # Input assertions
    assert os.path.exists(args.bible_dir), "Bible directory does not exist!"

    # Derive some paths
    en_bible_dir = pathlib.Path(os.path.join(args.bible_dir, "kj"))
    de_bible_dir = pathlib.Path(os.path.join(args.bible_dir, "guest"))

    # Create all (en, de) pairs
    books = {}
    files = list(en_bible_dir.glob("*/*.htm"))
    for en_fname in tqdm(files):

        # Pair name
        book_id, chapter_id = en_fname.parts[-2:]

        # Find german equivalent
        de_fname = os.path.join(de_bible_dir, book_id, chapter_id)

        # Parse both files
        en_book_name, en_content = parse_htm(en_fname)
        de_book_name, de_content = parse_htm(de_fname)

        # Add chapter
        if en_content and de_content:

            # Create book (if it does not exist already)
            if en_book_name not in books:
                books[en_book_name] = ParallelBook(en_book_name, de_book_name)

            books[en_book_name].add_parallel_chapter(chapter_id, en_content, de_content)

    # Print corpora & statistics
    print_corpora(books)
    print_statistics(books)

    # Save all verses
    save_verses(books,
                output_en_fname=args.output_en, output_de_fname=args.output_de,
                output_en_vocab_fname=args.output_en_vocab, output_de_vocab_fname=args.output_de_vocab)
    save_verses_as_tf_records(books, output_fname=args.output_tf)


if __name__ == "__main__":
    main()

import argparse
import sys

from data import build_vocabulary, create_training_pairs, load_corpus
from validation import run_checks


def configure_utf8_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the skip-gram correctness checks.")
    parser.add_argument("--corpus", default="corpus.txt")
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--checks", action="store_true", help="run all validation checks")
    return parser.parse_args()


def main():
    configure_utf8_console()
    args = parse_arguments()
    if not args.checks:
        print("Use --checks to run the validation suite.")
        return 0

    sentences = load_corpus(args.corpus)
    word_to_id, id_to_word = build_vocabulary(sentences)
    pairs = create_training_pairs(sentences, word_to_id, args.window)
    return 0 if run_checks(sentences, word_to_id, id_to_word, pairs) else 1


if __name__ == "__main__":
    sys.exit(main())

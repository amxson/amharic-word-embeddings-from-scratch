import argparse
import sys

from data import build_vocabulary, corpus_statistics, create_training_pairs, load_corpus
from skipgram import initialize_parameters, nearest_neighbors, train
from validation import run_checks

CORPUS_PATH = "corpus.txt"
WINDOW_SIZE = 2
EMBEDDING_DIM = 12
LEARNING_RATE = 0.05
EPOCHS = 30
SEED = 2026
QUERY_WORDS = ["ተማሪ", "አስተማሪ", "ቡና", "ገበሬ", "ዝናብ"]


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Amharic skip-gram embeddings from scratch.")
    parser.add_argument("--checks", action="store_true", help="run validation checks only")
    args = parser.parse_args()

    sentences = load_corpus(CORPUS_PATH)
    word_to_id, id_to_word = build_vocabulary(sentences)
    pairs = create_training_pairs(sentences, word_to_id, WINDOW_SIZE)
    if not pairs:
        parser.error("the corpus must contain at least one sentence with two words")

    if not args.checks:
        stats = corpus_statistics(sentences, id_to_word, pairs)
        V = len(id_to_word)
        print("=== CORPUS AND MODEL ===")
        print(f"Amharic sentences: {stats['sentence_count']}; tokens: {stats['token_count']}")
        print(f"Vocabulary: {V}; window: {WINDOW_SIZE}; training pairs: {len(pairs)}")
        print(f"V={V}, d={EMBEDDING_DIM}; E: {V} x {EMBEDDING_DIM}; U: {EMBEDDING_DIM} x {V}")
        print(f"Parameters: {2 * V * EMBEDDING_DIM}; seed: {SEED}")
        print(f"Learning rate: {LEARNING_RATE}; epochs: {EPOCHS}\n")

    if not run_checks(sentences, word_to_id, id_to_word, pairs):
        print("Validation failed; training was not started.")
        return 1
    if args.checks:
        return 0

    E, U = initialize_parameters(len(id_to_word), EMBEDDING_DIM, SEED)
    print("\n=== TRAINING ===")
    train(E, U, pairs, LEARNING_RATE, EPOCHS, SEED)

    print("\n=== NEAREST NEIGHBORS FROM E ===")
    for word in QUERY_WORDS:
        neighbors = nearest_neighbors(word, E, word_to_id, id_to_word, 3)
        if neighbors:
            formatted = ", ".join(f"{neighbor} ({score:.4f})" for neighbor, score in neighbors)
            print(f"{word}: {formatted}")
        else:
            print(f"{word}: unknown word")
    print("Lower loss alone does not prove semantic quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

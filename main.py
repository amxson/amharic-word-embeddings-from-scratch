import argparse
import sys

from data import build_vocabulary, corpus_statistics, create_training_pairs, load_corpus
from skipgram import initialize_parameters, nearest_neighbors, save_model, train, train_fresh_model
from validation import run_checks, verify_save_reload

CORPUS_PATH = "corpus.txt"
MODEL_PATH = "model.json"
WINDOW_SIZE = 2
EMBEDDING_DIM = 12
LEARNING_RATE = 0.05
EPOCHS = 30
SEED = 2026
QUERY_WORDS = ["ተማሪ", "አስተማሪ", "ቡና", "ገበሬ", "ዝናብ"]


def print_neighbors(E, word_to_id, id_to_word, words, heading):
    print(f"\n=== {heading} ===")
    for word in words:
        if word not in word_to_id:
            print(f"{word}: unknown word")
            continue
        neighbors = nearest_neighbors(word, E, word_to_id, id_to_word, 3)
        formatted = ", ".join(f"{neighbor} ({score:.4f})" for neighbor, score in neighbors)
        print(f"{word}: {formatted or 'no other vocabulary words'}")


def run_experiment(sentences, word_to_id, id_to_word, E, pairs, history):
    comparison_window = 1 if WINDOW_SIZE != 1 else 2
    comparison_pairs = create_training_pairs(sentences, word_to_id, comparison_window)
    print("\n=== CONTROLLED EXPERIMENT: CONTEXT WINDOW ONLY ===")
    print("Held fixed: corpus, preprocessing, vocabulary, dimension, learning rate, epochs, and seed")
    comparison_E, _, comparison_history = train_fresh_model(
        len(id_to_word), EMBEDDING_DIM, comparison_pairs, LEARNING_RATE, EPOCHS, SEED
    )
    print(f"Window {WINDOW_SIZE}: pairs={len(pairs)}, final mean loss={history[-1]:.6f}")
    print(f"Window {comparison_window}: pairs={len(comparison_pairs)}, final mean loss={comparison_history[-1]:.6f}")
    words = ["ተማሪ", "ቡና", "ገበሬ"]
    print_neighbors(E, word_to_id, id_to_word, words, f"WINDOW {WINDOW_SIZE} NEIGHBORS")
    print_neighbors(comparison_E, word_to_id, id_to_word, words, f"WINDOW {comparison_window} NEIGHBORS")
    print("The losses average different pair datasets; lower loss does not prove a window is universally better.")


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
    history = train(E, U, pairs, LEARNING_RATE, EPOCHS, SEED)

    print_neighbors(E, word_to_id, id_to_word, QUERY_WORDS, "FIVE NEAREST-NEIGHBOR QUERIES FROM E")
    print("Lower loss alone does not prove semantic quality.")

    settings = {
        "language": "Amharic",
        "context_window": WINDOW_SIZE,
        "embedding_dimension": EMBEDDING_DIM,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "seed": SEED,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "preprocessing": "NFC normalization; Unicode punctuation/symbol separation; whitespace tokenization",
    }
    print("\n=== REAL-MODEL JSON SAVE/RELOAD ===")
    save_model(MODEL_PATH, id_to_word, E, U, settings)
    query = next((word for word in QUERY_WORDS if word in word_to_id), id_to_word[0])
    if not verify_save_reload(MODEL_PATH, id_to_word, E, U, settings, query):
        return 1

    run_experiment(sentences, word_to_id, id_to_word, E, pairs, history)
    return 0


if __name__ == "__main__":
    sys.exit(main())

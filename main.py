import argparse
import math
import sys

from data import build_vocabulary, corpus_statistics, create_training_pairs, load_corpus
from skipgram import initialize_parameters, load_model, nearest_neighbors, save_model, train
from validation import run_checks, verify_save_reload

DEFAULT_QUERIES = ["ተማሪ", "አስተማሪ", "ቡና", "ገበሬ", "ዝናብ"]


def configure_utf8_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train, verify, compare, save, and query Amharic skip-gram embeddings."
    )
    parser.add_argument("--corpus", default="corpus.txt")
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", default="model.json")
    parser.add_argument("--load-model", help="load JSON instead of training")
    parser.add_argument("--checks", action="store_true", help="run checks only")
    parser.add_argument("--check-data", action="store_true", help="show corpus statistics only")
    parser.add_argument("--show-evolution", action="store_true")
    parser.add_argument("--skip-experiment", action="store_true")
    parser.add_argument("--no-interactive", action="store_true")
    args = parser.parse_args()
    if min(args.window, args.dim, args.epochs) < 1:
        parser.error("window, dimension, and epochs must be positive integers")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("learning rate must be finite and positive")
    return args


def print_corpus_information(statistics, window):
    print("=== AMHARIC CORPUS ===")
    print(f"Sentence count: {statistics['sentence_count']}")
    print(f"Token count: {statistics['token_count']}")
    print(f"Vocabulary size: {statistics['vocabulary_size']}")
    print(f"Context window: {window}")
    print(f"Training-pair count: {statistics['training_pair_count']}")
    print("Sentence boundaries preserved: yes")


def print_neighbors(E, word_to_id, id_to_word, queries, heading):
    print(f"\n=== {heading} ===")
    for query in queries:
        neighbors = nearest_neighbors(query, E, word_to_id, id_to_word, 3)
        if not neighbors:
            print(f"{query}: unknown word")
        else:
            formatted = ", ".join(f"{word} ({score:.4f})" for word, score in neighbors)
            print(f"{query}: {formatted}")


def interactive_query(E, word_to_id, id_to_word):
    print("\nType an Amharic word for its neighbors; type q to quit.")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query.lower() in {"q", "quit", "exit"}:
            return
        neighbors = nearest_neighbors(query, E, word_to_id, id_to_word, 3)
        if not neighbors:
            print(f"Unknown word: {query}")
        else:
            print(", ".join(f"{word} ({score:.4f})" for word, score in neighbors))


def load_and_query(path, no_interactive):
    vocabulary, E, U, settings = load_model(path)
    word_to_id = {word: index for index, word in enumerate(vocabulary)}
    print("=== LOADED MODEL ===")
    print(f"Vocabulary size: {len(vocabulary)}; E: {len(E)} x {len(E[0])}; U: {len(U)} x {len(U[0])}")
    print(f"Settings: {settings}")
    print_neighbors(E, word_to_id, vocabulary, DEFAULT_QUERIES, "NEAREST NEIGHBORS FROM E")
    if not no_interactive:
        interactive_query(E, word_to_id, vocabulary)


def main():
    configure_utf8_console()
    args = parse_arguments()
    if args.load_model:
        load_and_query(args.load_model, args.no_interactive)
        return 0

    sentences = load_corpus(args.corpus)
    word_to_id, id_to_word = build_vocabulary(sentences)
    pairs = create_training_pairs(sentences, word_to_id, args.window)
    statistics = corpus_statistics(sentences, id_to_word, pairs)
    print_corpus_information(statistics, args.window)
    if args.check_data:
        return 0
    if not pairs:
        print("The corpus must contain at least one sentence with two words.")
        return 1

    if not run_checks(sentences, word_to_id, id_to_word, pairs):
        print("Validation failed; real-model training was not started.")
        return 1
    if args.checks:
        return 0

    print("\n=== MODEL AND TRAINING SETTINGS ===")
    print(f"Random seed: {args.seed}")
    print(f"V={len(id_to_word)}, d={args.dim}")
    print(f"E shape: {len(id_to_word)} x {args.dim}; U shape: {args.dim} x {len(id_to_word)}")
    print(f"Total parameters: {2 * len(id_to_word) * args.dim}")
    print(f"Window={args.window}, learning rate={args.learning_rate}, epochs={args.epochs}")

    E, U = initialize_parameters(len(id_to_word), args.dim, args.seed)
    initial_E = [row[:] for row in E]
    print("\n=== FULL-SOFTMAX SGD ===")
    history = train(E, U, pairs, args.learning_rate, args.epochs, args.seed, verbose=True)

    if args.show_evolution:
        print_neighbors(initial_E, word_to_id, id_to_word, ["ተማሪ", "ቡና", "ዝናብ"], "BEFORE TRAINING (RANDOM; NO SEMANTIC MEANING)")
        print_neighbors(E, word_to_id, id_to_word, ["ተማሪ", "ቡና", "ዝናብ"], "AFTER TRAINING")

    print_neighbors(E, word_to_id, id_to_word, DEFAULT_QUERIES, "FIVE AMHARIC QUERY WORDS")
    print("Note: lower loss alone does not prove semantic quality.")

    settings = {
        "language": "Amharic",
        "context_window": args.window,
        "embedding_dimension": args.dim,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "seed": args.seed,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "preprocessing": "NFC normalization; Unicode punctuation separation; whitespace tokenization",
    }
    save_model(args.model, id_to_word, E, U, settings)
    query = next((word for word in DEFAULT_QUERIES if word in word_to_id), id_to_word[0])
    if not verify_save_reload(args.model, id_to_word, E, U, settings, query):
        return 1

    if not args.skip_experiment:
        comparison_window = 1 if args.window != 1 else 2
        comparison_pairs = create_training_pairs(sentences, word_to_id, comparison_window)
        comparison_E, comparison_U = initialize_parameters(len(id_to_word), args.dim, args.seed)
        print("\n=== CONTROLLED EXPERIMENT: CONTEXT WINDOW ONLY ===")
        print("Held fixed: corpus, preprocessing, seed, d, learning rate, and epochs")
        comparison_history = train(
            comparison_E,
            comparison_U,
            comparison_pairs,
            args.learning_rate,
            args.epochs,
            args.seed,
            verbose=False,
        )
        print(f"Window {args.window}: pairs={len(pairs)}, final mean loss={history[-1]:.6f}")
        print(f"Window {comparison_window}: pairs={len(comparison_pairs)}, final mean loss={comparison_history[-1]:.6f}")
        for query in ["ተማሪ", "ቡና", "ገበሬ"]:
            main_words = [word for word, _ in nearest_neighbors(query, E, word_to_id, id_to_word)]
            comparison_words = [word for word, _ in nearest_neighbors(query, comparison_E, word_to_id, id_to_word)]
            print(f"{query}: window {args.window}={main_words}; window {comparison_window}={comparison_words}")
        print("The losses average different pair sets, so this is descriptive rather than a universal ranking.")

    if not args.no_interactive:
        interactive_query(E, word_to_id, id_to_word)
    return 0


if __name__ == "__main__":
    sys.exit(main())

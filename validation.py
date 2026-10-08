import math
import tempfile
from pathlib import Path

from data import build_vocabulary, create_training_pairs
from skipgram import (
    compute_gradients,
    cosine_similarity,
    forward,
    load_model,
    nearest_neighbors,
    prediction_probabilities,
    save_model,
    stable_softmax_and_loss,
    train_fresh_model,
    train_pair,
)

TOLERANCE = 0.00001


def close(actual, expected, tolerance=TOLERANCE):
    return abs(actual - expected) <= tolerance


def vector_close(actual, expected, tolerance=TOLERANCE):
    return len(actual) == len(expected) and all(close(a, e, tolerance) for a, e in zip(actual, expected))


def matrix_close(actual, expected, tolerance=TOLERANCE):
    return len(actual) == len(expected) and all(vector_close(a, e, tolerance) for a, e in zip(actual, expected))


def report(passed, label, details=""):
    suffix = f" - {details}" if details else ""
    print(f"{'PASS' if passed else 'FAIL'}: {label}{suffix}")
    return passed


def mandatory_checks():
    results = []
    words = ["a", "b", "c", "d"]
    mapping = {word: index for index, word in enumerate(words)}
    pairs = create_training_pairs([words], mapping, 1)
    expected_pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    cross_pairs = create_training_pairs([["a"], ["b"]], mapping, 1)
    results.append(report(pairs == expected_pairs and cross_pairs == [], "1 context window", "expected pairs and no cross-sentence pairs"))

    E = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    U = [[0.0, 1.0, -1.0], [1.0, 0.0, 1.0]]
    h, scores, probabilities, loss = forward(E, U, 0, 1)
    results.append(report(vector_close(scores, [0.0, 1.0, -1.0]), "2 forward pass", f"scores={scores}"))

    expected_probabilities = [0.244728, 0.665241, 0.090031]
    probability_ok = vector_close(probabilities, expected_probabilities) and close(loss, 0.407606)
    probability_ok = probability_ok and all(value >= 0.0 for value in probabilities) and close(sum(probabilities), 1.0)
    results.append(report(probability_ok, "3 probabilities and loss", f"p={[round(x, 6) for x in probabilities]}, L={loss:.6f}"))

    grad_U, grad_h = compute_gradients(h, U, probabilities, 1)
    expected_grad_U = [[0.244728, -0.334759, 0.090031], [0.0, 0.0, 0.0]]
    expected_grad_h = [-0.424790, 0.334759]
    results.append(report(matrix_close(grad_U, expected_grad_U) and vector_close(grad_h, expected_grad_h), "4 gradients", f"grad_h={[round(x, 6) for x in grad_h]}"))

    updated_E = [row[:] for row in E]
    updated_U = [row[:] for row in U]
    train_pair(updated_E, updated_U, 0, 1, 0.1)
    _, _, _, updated_loss = forward(updated_E, updated_U, 0, 1)
    unchanged = updated_E[1:] == E[1:]
    update_ok = vector_close(updated_E[0], [1.042479, -0.033476]) and close(updated_loss, 0.361859) and unchanged
    results.append(report(update_ok, "5 one update", f"E[cats]={[round(x, 6) for x in updated_E[0]]}, L={updated_loss:.6f}"))

    extreme_probabilities, extreme_loss = stable_softmax_and_loss([1000.0, 0.0], 1)
    extreme_ok = math.isfinite(extreme_loss) and close(extreme_loss, 1000.0)
    results.append(report(extreme_ok, "6 extreme scores", f"p={extreme_probabilities}, L={extreme_loss:.6f}"))
    print(f"Mandatory checks passed: {sum(results)}/6")
    return all(results)


def shift_invariance_check():
    probabilities, loss = stable_softmax_and_loss([0.0, 1.0, -1.0], 1)
    shifted_probabilities, shifted_loss = stable_softmax_and_loss([100.0, 101.0, 99.0], 1)
    passed = vector_close(probabilities, shifted_probabilities) and close(loss, shifted_loss)
    return report(passed, "optional shift invariance", "probabilities and loss unchanged after +100")


def numerical_gradient_check():
    epsilon = 0.00001
    E = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    U = [[0.0, 1.0, -1.0], [1.0, 0.0, 1.0]]
    h, _, probabilities, _ = forward(E, U, 0, 1)
    grad_U, grad_h = compute_gradients(h, U, probabilities, 1)
    samples = []

    for k, j in [(0, 0), (0, 1), (1, 2)]:
        original = U[k][j]
        U[k][j] = original + epsilon
        plus = forward(E, U, 0, 1)[3]
        U[k][j] = original - epsilon
        minus = forward(E, U, 0, 1)[3]
        U[k][j] = original
        numerical = (plus - minus) / (2.0 * epsilon)
        samples.append((f"U[{k},{j}]", grad_U[k][j], numerical))

    for k in [0, 1]:
        original = E[0][k]
        E[0][k] = original + epsilon
        plus = forward(E, U, 0, 1)[3]
        E[0][k] = original - epsilon
        minus = forward(E, U, 0, 1)[3]
        E[0][k] = original
        numerical = (plus - minus) / (2.0 * epsilon)
        samples.append((f"E[0,{k}]", grad_h[k], numerical))

    print("Numerical-gradient samples:")
    errors = []
    for name, analytical, numerical in samples:
        error = abs(analytical - numerical)
        errors.append(error)
        print(f"  {name}: analytical={analytical:.9f}, numerical={numerical:.9f}, error={error:.3e}")
    maximum_error = max(errors)
    passed = maximum_error <= TOLERANCE
    report(passed, "numerical gradient", f"epsilon={epsilon:.5f}, maximum error={maximum_error:.3e}")
    return passed, maximum_error


def robustness_checks():
    zero_ok = cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    unknown_ok = nearest_neighbors("unknown", [[1.0, 0.0]], {"known": 0}, ["known"]) == []
    return [
        report(zero_ok, "zero-norm cosine", "returns 0.0 without division by zero"),
        report(unknown_ok, "unknown query", "returns an empty result without crashing"),
    ]


def save_reload_check():
    vocabulary = ["a", "b", "c"]
    word_to_id = {word: index for index, word in enumerate(vocabulary)}
    E = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    U = [[0.0, 1.0, -1.0], [1.0, 0.0, 1.0]]
    settings = {"dimension": 2, "seed": 7}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        save_model(path, vocabulary, E, U, settings)
        loaded_vocabulary, loaded_E, loaded_U, loaded_settings = load_model(path)
    loaded_mapping = {word: index for index, word in enumerate(loaded_vocabulary)}
    passed = loaded_vocabulary == vocabulary and loaded_settings == settings
    passed = passed and vector_close(prediction_probabilities(E, U, 0), prediction_probabilities(loaded_E, loaded_U, 0))
    passed = passed and nearest_neighbors("a", E, word_to_id, vocabulary) == nearest_neighbors("a", loaded_E, loaded_mapping, loaded_vocabulary)
    return report(passed, "save/reload", "vocabulary, settings, probabilities, and neighbor order match")


def reproducibility_check(sentences, word_to_id, id_to_word, pairs):
    arguments = (len(id_to_word), 4, pairs, 0.03, 2, 2026)
    E1, U1, history1 = train_fresh_model(*arguments)
    E2, U2, history2 = train_fresh_model(*arguments)
    queries = [word for word in ["ተማሪ", "ቡና", "ገበሬ"] if word in word_to_id]
    orders1 = [nearest_neighbors(word, E1, word_to_id, id_to_word) for word in queries]
    orders2 = [nearest_neighbors(word, E2, word_to_id, id_to_word) for word in queries]
    passed = E1 == E2 and U1 == U2 and history1 == history2 and orders1 == orders2
    return report(passed, "reproducibility", "same seed/configuration produced identical losses, parameters, and neighbors")


def verify_save_reload(path, vocabulary, E, U, settings, query):
    loaded_vocabulary, loaded_E, loaded_U, loaded_settings = load_model(path)
    mapping = {word: index for index, word in enumerate(vocabulary)}
    loaded_mapping = {word: index for index, word in enumerate(loaded_vocabulary)}
    passed = loaded_vocabulary == vocabulary and loaded_settings == settings
    passed = passed and vector_close(prediction_probabilities(E, U, mapping[query]), prediction_probabilities(loaded_E, loaded_U, loaded_mapping[query]))
    passed = passed and nearest_neighbors(query, E, mapping, vocabulary) == nearest_neighbors(query, loaded_E, loaded_mapping, loaded_vocabulary)
    return report(passed, "trained-model save/reload", "real-model predictions and neighbor order match")


def run_checks(sentences, word_to_id, id_to_word, pairs):
    print("=== VALIDATION CHECKS ===")
    passed = mandatory_checks()
    print("\n=== OPTIONAL AND ROBUSTNESS CHECKS ===")
    passed = shift_invariance_check() and passed
    gradient_ok, _ = numerical_gradient_check()
    passed = gradient_ok and passed
    passed = all(robustness_checks()) and passed
    passed = save_reload_check() and passed
    passed = reproducibility_check(sentences, word_to_id, id_to_word, pairs) and passed
    return passed

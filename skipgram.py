import math
import random


def initialize_parameters(vocabulary_size, dimension, seed):
    rng = random.Random(seed)
    scale = 0.5 / dimension
    E = [
        [rng.uniform(-scale, scale) for _ in range(dimension)]
        for _ in range(vocabulary_size)
    ]
    U = [
        [rng.uniform(-scale, scale) for _ in range(vocabulary_size)]
        for _ in range(dimension)
    ]
    return E, U


def stable_softmax_and_loss(scores, target_id):
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    denominator = sum(exponentials)
    probabilities = [value / denominator for value in exponentials]
    loss = (maximum - scores[target_id]) + math.log(denominator)
    return probabilities, loss


def forward(E, U, center_id, target_id):
    h = E[center_id][:]
    vocabulary_size = len(U[0])
    scores = []
    for output_id in range(vocabulary_size):
        score = 0.0
        for k in range(len(h)):
            score += h[k] * U[k][output_id]
        scores.append(score)
    probabilities, loss = stable_softmax_and_loss(scores, target_id)
    return h, scores, probabilities, loss


def compute_gradients(h, U, probabilities, target_id):
    error = probabilities[:]
    error[target_id] -= 1.0
    dimension = len(h)
    vocabulary_size = len(error)

    grad_U = [[0.0 for _ in range(vocabulary_size)] for _ in range(dimension)]
    grad_h = [0.0 for _ in range(dimension)]
    for k in range(dimension):
        for output_id in range(vocabulary_size):
            grad_U[k][output_id] = h[k] * error[output_id]
            grad_h[k] += U[k][output_id] * error[output_id]
    return grad_U, grad_h


def train_pair(E, U, center_id, target_id, learning_rate):
    h, _, probabilities, loss = forward(E, U, center_id, target_id)

    # Both gradients are calculated from the old forward-pass weights.
    grad_U, grad_h = compute_gradients(h, U, probabilities, target_id)
    for k in range(len(U)):
        for output_id in range(len(U[k])):
            U[k][output_id] -= learning_rate * grad_U[k][output_id]
    for k in range(len(E[center_id])):
        E[center_id][k] -= learning_rate * grad_h[k]
    return loss


def dataset_loss(E, U, pairs):
    """Evaluate all pairs at one fixed set of weights without updating."""
    total = 0.0
    for center_id, target_id in pairs:
        _, _, _, loss = forward(E, U, center_id, target_id)
        total += loss
    return total / len(pairs)


def train(E, U, pairs, learning_rate, epochs, seed, verbose=True):
    rng = random.Random(seed)
    initial_loss = dataset_loss(E, U, pairs)
    history = [initial_loss]
    if verbose:
        print(f"Initial mean loss: {initial_loss:.6f}")

    order = list(range(len(pairs)))
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        for pair_index in order:
            center_id, target_id = pairs[pair_index]
            train_pair(E, U, center_id, target_id, learning_rate)
        mean_loss = dataset_loss(E, U, pairs)
        history.append(mean_loss)
        if verbose:
            print(f"Epoch {epoch} mean loss: {mean_loss:.6f}")

    if verbose:
        decrease = 100.0 * (initial_loss - history[-1]) / initial_loss
        print(f"Final mean loss: {history[-1]:.6f}")
        print(f"Loss decrease: {decrease:.2f}%")
    return history

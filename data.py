import unicodedata
from pathlib import Path


def preprocess_sentence(text):
    normalized = unicodedata.normalize("NFC", text.strip())
    separated = []
    for character in normalized:
        category = unicodedata.category(character)
        separated.append(" " if category[0] in {"P", "S"} else character)
    return [
        token
        for token in "".join(separated).split()
        if any(unicodedata.category(character).startswith("L") for character in token)
    ]


def load_corpus(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [tokens for line in lines if (tokens := preprocess_sentence(line))]


def build_vocabulary(sentences):
    id_to_word = sorted({word for sentence in sentences for word in sentence})
    word_to_id = {word: index for index, word in enumerate(id_to_word)}
    return word_to_id, id_to_word


def create_training_pairs(sentences, word_to_id, window_size):
    pairs = []
    for sentence in sentences:
        ids = [word_to_id[word] for word in sentence]
        for center_position, center_id in enumerate(ids):
            left = max(0, center_position - window_size)
            right = min(len(ids), center_position + window_size + 1)
            for context_position in range(left, right):
                if context_position != center_position:
                    pairs.append((center_id, ids[context_position]))
    return pairs


def corpus_statistics(sentences, vocabulary, pairs):
    return {
        "sentence_count": len(sentences),
        "token_count": sum(len(sentence) for sentence in sentences),
        "vocabulary_size": len(vocabulary),
        "training_pair_count": len(pairs),
    }

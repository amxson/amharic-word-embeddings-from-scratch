# Amharic Word Embeddings from Scratch

This project trains small Amharic word embeddings with a skip-gram model written entirely with the Python standard library. Matrix operations, stable softmax, cross-entropy loss, analytical gradients, stochastic gradient descent, cosine similarity, and numerical gradient checking use explicit lists and loops.

Word embeddings represent words as dense vectors that can be compared or used as inputs to later language models. Skip-gram learns these vectors by predicting surrounding context words from a center word.

## Project structure

```text
data.py       corpus loading, preprocessing, vocabulary, and training pairs
skipgram.py   full-softmax model, manual gradients, SGD, similarity, and JSON persistence
validation.py assignment checks, numerical verification, and reproducibility
main.py       complete assignment workflow and checks-only mode
corpus.txt    one Amharic sentence per line
```

## Data preparation

The local corpus contains short, understandable Amharic sentences with repeated patterns in education, family, food, farming, weather, health, transport, places, language, and music. Preprocessing applies Unicode NFC normalization, separates punctuation and symbols, tokenizes on whitespace, and retains word tokens. It deliberately avoids stemming or aggressive spelling normalization. Vocabulary IDs are deterministic because unique words are sorted. Training pairs are generated independently within each sentence, so they never cross sentence boundaries.

| Statistic at window 2 | Value |
|---|---:|
| Sentences | 85 |
| Tokens | 316 |
| Vocabulary size | 200 |
| Directed training pairs | 754 |

## Skip-gram mathematics

For vocabulary size `V` and dimension `d`, the input embeddings are `E` with shape `V x d`, and output parameters are `U` with shape `d x V`. For center index `i` and context target `t`:

```text
h = E[i]
s = hU
m = max(s)
a[j] = exp(s[j] - m)
p[j] = a[j] / sum(a)
L = (m - s[t]) + ln(sum(a))
```

The error is `e[j] = p[j] - 1` at the target and `p[j]` otherwise:

```text
grad_U[k,j] = h[k] * e[j]
grad_h[k]   = sum_j U[k,j] * e[j]
U          = U - learning_rate * grad_U
E[i]       = E[i] - learning_rate * grad_h
```

Both gradients are computed from the old forward-pass weights before either matrix is updated. Each pair updates all of `U` and only the selected row `E[i]`. Pair order is shuffled reproducibly every epoch. Dataset loss is evaluated separately over all pairs at a single fixed set of weights.

## What makes the implementation verifiable?

- Six exact assignment checks at tolerance `0.00001` cover pairs, scores, probabilities and loss, gradients, one update (including every entry of `U` and unchanged other rows of `E`), and extreme scores.
- Central differences with `epsilon = 0.00001` verify representative entries of `U` and `E[i]`; the recorded maximum error is `5.330e-12`.
- Shift invariance, zero-norm cosine, and unknown queries are checked.
- Fixed-weight loss evaluation must leave both matrices unchanged, even when repeated.
- Two same-seed training runs on the real corpus (`d=4`, two epochs) must produce identical initialization, losses, parameters, and neighbor order.
- JSON reload must preserve vocabulary, settings, probabilities, and neighbor order.
- Real-model training is blocked if a validation check fails.

## Training and interpretation

The configuration constants near the top of `main.py` use seed `2026`, window `2`, dimension `12`, learning rate `0.05`, and `30` epochs. Change those constants directly to adjust training. The program prints corpus and model statistics, initial fixed-weight loss, loss after every epoch, final loss, and percentage reduction. It reports three nearest neighbors for five Amharic query words using manually implemented cosine similarity.

Both matrices start with small uniform random values in `[-0.5/d, 0.5/d]`. The 12-dimensional model has only `2Vd = 4,800` parameters; this keeps full softmax practical. The learning rate and 30 epochs give a stable decreasing loss curve and observable contextual relationships without a large search framework. Cosine similarity uses rows of `E`, excludes the query itself, and returns zero safely for a zero-norm vector.

With the committed corpus and settings, mean loss decreases from `5.298295` to `2.616722`, a `50.61%` reduction. Example trained neighbors are:

| Query | Top three neighbors from `E` |
|---|---|
| `ተማሪ` | `በየቀኑ`, `ይጠይቃል`, `ቃላትን` |
| `አስተማሪ` | `ያስረዳል`, `ይጠይቃል`, `ይመለከታል` |
| `ቡና` | `ትቀርባለች`, `ሻይ`, `ትፈላለች` |
| `ገበሬ` | `ስንዴ`, `በዝናብ`, `ይሠራል` |
| `ዝናብ` | `ያቀርባል`, `ይረጥባል`, `መሬት` |

## Model persistence and controlled experiment

After training, the normal run saves the real Amharic vocabulary, `E`, `U`, and settings to `model.json`. It reloads the model and verifies exact vocabulary/settings equality, prediction probabilities within tolerance `0.00001`, and identical nearest-neighbor ordering. The small synthetic round-trip check also remains part of validation.

The normal run then changes only context window 2 to window 1. Corpus, preprocessing, vocabulary, dimension, learning rate, epochs, and seed stay fixed. A fresh comparison model is trained with the same initialization settings.

| Window | Directed pairs | Final mean loss |
|---:|---:|---:|
| 2 | 754 | 2.616722 |
| 1 | 462 | 3.358973 |

| Query | Window 2 neighbors | Window 1 neighbors |
|---|---|---|
| `ተማሪ` | `በየቀኑ`, `ይጠይቃል`, `ቃላትን` | `ይጠይቃል`, `ያጠናል`, `አስተማሪ` |
| `ቡና` | `ትቀርባለች`, `ሻይ`, `ትፈላለች` | `ትፈላለች`, `ሻይ`, `ያሳድጋል` |
| `ገበሬ` | `ስንዴ`, `በዝናብ`, `ይሠራል` | `ይዘራል`, `በዝናብ`, `ይሰበስባል` |

These losses average different pair datasets. The smaller value alone does not prove that one context window is universally better; the neighbor comparison shows how changing local context changes the learned representation.

## Run

Python 3.8 or newer is sufficient; there are no external dependencies.

```bash
python main.py
python main.py --checks
```

`python main.py` runs validation, training, five neighbor queries, real-model JSON save/reload verification, and the controlled context-window experiment. `python main.py --checks` runs the validation suite only, including shift invariance, synthetic save/reload, numerical gradients, reproducibility, fixed-weight evaluation, and unknown/zero-norm safety. Training settings are edited through the constants in `main.py`; no other run modes are needed. Generated `model.json` files are ignored by Git.

## Limitations

This intentionally tiny corpus and full-softmax implementation prioritize mathematical clarity over scale. Amharic morphology is not analyzed, sparse surface forms remain, and frequent grammatical contexts can dominate neighbors. The results demonstrate reproducible contextual learning; lower loss alone does not prove broad semantic quality or production readiness.

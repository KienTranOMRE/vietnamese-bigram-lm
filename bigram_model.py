
import math
import re
import random
from collections import defaultdict
from datasets import load_dataset


# 1. Load Vietnamese Wikipedia dataset
def load_corpus(num_articles=500):
    """Load Vietnamese Wikipedia articles from HuggingFace datasets."""
    dataset = load_dataset(
        "wikimedia/wikipedia", "20231101.vi",
        split=f"train[:{num_articles}]",
    )
    corpus = "\n".join(article["text"] for article in dataset)
    print(f"Đã tải {num_articles} bài viết, tổng {len(corpus):,} ký tự.")
    return corpus


# 2. Preprocessing
def preprocess(text):
    """Lowercase, split into sentences, tokenize into syllables."""
    text = text.lower().strip()
    # Remove references: [1], [2], etc.
    text = re.sub(r'\[.*?\]', '', text)
    # Remove special characters 
    text = re.sub(r'[(){}"""\'\-–—:;,/\\|@#$%^&*+=~`<>]', ' ', text)
    # Split into sentences by punctuation and newlines
    sentences = re.split(r'[.!?\n]+', text)
    tokenized = []
    for sent in sentences:
        syllables = sent.split()
        if len(syllables) >= 2:  # Skip very short fragments
            tokenized.append(['<s>'] + syllables + ['</s>'])
    return tokenized


# 3. Build bigram model
def build_bigram_model(sentences):
    """Build bigram counts and unigram counts."""
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    for sent in sentences:
        for i in range(len(sent) - 1):
            bigram_counts[(sent[i], sent[i + 1])] += 1
            unigram_counts[sent[i]] += 1
        unigram_counts[sent[-1]] += 1

    return bigram_counts, unigram_counts


def bigram_probability(w1, w2, bigram_counts, unigram_counts, vocab_size=0):
    """P(w2 | w1) with optional Laplace smoothing.
    If vocab_size > 0, apply add-1 (Laplace) smoothing.
    """
    if vocab_size > 0:
        return (bigram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + vocab_size)
    if unigram_counts[w1] == 0:
        return 0.0
    return bigram_counts[(w1, w2)] / unigram_counts[w1]


# 4. Sentence probability
def sentence_probability(sentence_str, bigram_counts, unigram_counts, vocab_size=0):
    """Calculate joint probability of a sentence using the bigram model."""
    syllables = sentence_str.lower().split()
    tokens = ['<s>'] + syllables + ['</s>']

    prob = 1.0
    details = []
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        p = bigram_probability(w1, w2, bigram_counts, unigram_counts, vocab_size)
        details.append((w1, w2, p))
        prob *= p

    return prob, details


# 5. Text generation
def generate_sentence(bigram_counts, max_len=20):
    """Generate a sentence by sampling from bigram probabilities."""
    current = '<s>'
    result = []

    for _ in range(max_len):
        # Get all possible next words
        candidates = []
        weights = []
        for (w1, w2), count in bigram_counts.items():
            if w1 == current:
                candidates.append(w2)
                weights.append(count)

        if not candidates:
            break

        next_word = random.choices(candidates, weights=weights, k=1)[0]

        if next_word == '</s>':
            break

        result.append(next_word)
        current = next_word

    return ' '.join(result)


def main():
    print("=" * 60)
    print("VIETNAMESE SYLLABLE-LEVEL BIGRAM LANGUAGE MODEL")
    print("=" * 60)

    # Load and preprocess
    corpus = load_corpus(num_articles=5000)
    sentences = preprocess(corpus)
    print(f"Số câu sau tiền xử lý: {len(sentences):,}")
    print(f"Ví dụ câu đã tokenize: {sentences[0]}")

    # Build model
    bigram_counts, unigram_counts = build_bigram_model(sentences)
    print(f"\nSố bigram khác nhau: {len(bigram_counts):,}")
    print(f"Số unigram khác nhau: {len(unigram_counts):,}")

    # Show top bigrams
    print("\n--- Top 20 Bigram (theo tần suất) ---")
    sorted_bigrams = sorted(bigram_counts.items(), key=lambda x: -x[1])
    for (w1, w2), count in sorted_bigrams[:20]:
        p = bigram_probability(w1, w2, bigram_counts, unigram_counts)
        print(f"  P({w2:12s} | {w1:12s}) = {count:5d}/{unigram_counts[w1]:<5d} = {p:.4f}")

    # Calculate probability for target sentence
    vocab_size = len(unigram_counts)
    target = "Hôm nay trời đẹp lắm"

    # MLE (no smoothing)
    print(f"\n--- Xác suất câu (MLE): \"{target}\" ---")
    prob, details = sentence_probability(target, bigram_counts, unigram_counts)
    for w1, w2, p in details:
        print(f"  P({w2:10s} | {w1:10s}) = {p:.6f}")
    print(f"\n  P(\"{target}\") = {prob:.10e}")

    log_prob = 0.0
    for _, _, p in details:
        if p > 0:
            log_prob += math.log2(p)
        else:
            log_prob = float('-inf')
            break
    print(f"  Log2 P = {log_prob:.4f}")

    # With Laplace smoothing
    print(f"\n--- Xác suất câu (Laplace smoothing, V={vocab_size:,}): \"{target}\" ---")
    prob_s, details_s = sentence_probability(target, bigram_counts, unigram_counts, vocab_size)
    for w1, w2, p in details_s:
        print(f"  P({w2:10s} | {w1:10s}) = {p:.6f}")
    print(f"\n  P(\"{target}\") = {prob_s:.10e}")

    log_prob_s = sum(math.log2(p) for _, _, p in details_s)
    print(f"  Log2 P = {log_prob_s:.4f}")

    # Compare two sentences
    target2 = "Việt Nam là một quốc gia"
    print(f"\n--- So sánh xác suất (Laplace smoothing) ---")
    for sent in [target, target2]:
        p, _ = sentence_probability(sent, bigram_counts, unigram_counts, vocab_size)
        lp = sum(math.log2(pp) for _, _, pp in _)
        print(f"  \"{sent}\":  P = {p:.6e},  Log2 P = {lp:.4f}")

    # Generate sentences
    print("\n--- Sinh câu từ mô hình bigram ---")
    random.seed(42)
    for i in range(5):
        generated = generate_sentence(bigram_counts)
        print(f"  [{i + 1}] {generated}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()

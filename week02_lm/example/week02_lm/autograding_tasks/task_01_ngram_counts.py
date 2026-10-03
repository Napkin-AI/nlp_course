"""Task 01: n-gram counting."""

from tqdm import tqdm
from collections import defaultdict, Counter

EOS = "_EOS_"
UNK = "_UNK_"


def count_ngrams(lines, n):
    """Count next-token frequencies for an n-gram language model.

    Args:
        lines: iterable of strings. Each string is a whitespace-tokenized line.
        n: n-gram order. Must be positive.

    Returns:
        A mapping from prefix tuples of length ``n - 1`` to mappings/counters
        of next-token counts. Each line should be treated as ending with EOS.
        Short prefixes should be left-padded with UNK.
    """
    if n <= 0:
        raise ValueError("n_must_be_positive")

    counts = defaultdict(Counter)
    # counts[(word1, word2)][word3] = how many times word3 occured after (word1, word2)
    for line in lines:
        tokens = [UNK] * (n - 1) + line.split() + [EOS]
        for start in range(n - 1, len(tokens)):
            counts[tuple(tokens[start - n + 1:start])][tokens[start]] += 1

    return counts

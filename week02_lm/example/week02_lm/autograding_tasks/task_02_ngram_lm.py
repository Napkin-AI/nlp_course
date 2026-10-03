"""Task 02: n-gram language model."""

from tqdm import tqdm
from collections import defaultdict, Counter

class NGramLanguageModel:

    def __init__(self, lines, n):
        """
        Train a simple count-based language model:
        compute probabilities P(w_t | prefix) given ngram counts

        :param n: computes probability of next token given (n - 1) previous words
        :param lines: an iterable of strings with space-separated tokens
        """
        if n <= 0:
            raise ValueError("test_invalid_n")

        self.n = n
        self.UNK, self.EOS = "_UNK_", "_EOS_"
        counts = self.count_ngrams(lines, self.n)

        # compute token proabilities given counts
        self.probs = defaultdict(Counter)
        # probs[(word1, word2)][word3] = P(word3 | word1, word2)

        # populate self.probs with actual probabilities
        for context, counter in counts.items():
            context_length = sum(counter.values())
            for word, cnt in counter.items():
                self.probs[context][word] = cnt / context_length


    def get_possible_next_tokens(self, prefix):
        """
        :param prefix: string with space-separated prefix tokens
        :returns: a dictionary {token : it's probability} for all tokens with positive probabilities
        """
        prefix = prefix.split()
        prefix = prefix[max(0, len(prefix) - self.n + 1):]
        prefix = [ self.UNK ] * (self.n - 1 - len(prefix)) + prefix
        return self.probs[tuple(prefix)]

    def get_next_token_prob(self, prefix, next_token):
        """
        :param prefix: string with space-separated prefix tokens
        :param next_token: the next token to predict probability for
        :returns: P(next_token|prefix) a single number, 0 <= P <= 1
        """
        return self.get_possible_next_tokens(prefix).get(next_token, 0)

    def count_ngrams(self, lines, n):
        """
        Count how many times each word occured after (n - 1) previous words
        :param lines: an iterable of strings with space-separated tokens
        :returns: a dictionary { tuple(prefix_tokens): {next_token_1: count_1, next_token_2: count_2}}

        When building counts, please consider the following two edge cases:
        - every prefix must contain exactly (n - 1) tokens. If a prefix is shorter than that,
        pad it with UNK on the left. The number of padding tokens therefore depends on n:
        derive it from n, do not hard-code it. For n=3,
        empty prefix: "" -> (UNK, UNK)
        short prefix: "the" -> (UNK, the)
        long prefix: "the new approach" -> (new, approach)
        For n=2 the very same prefixes give (UNK,), (the,) and (approach,), and for n=1 all
        of them give the empty tuple (). Equivalently: prepend (n - 1) UNK tokens to a line,
        then slide a window of size n over it.
        - you should add a special token, EOS, at the end of each sequence
        "... with deep neural networks ." -> (..., with, deep, neural, networks, ., EOS)
        count the probability of this token just like all others.

        Useful invariant: a line with L tokens always contributes exactly L + 1 counted events
        (its L tokens plus EOS), for every n. Note also that NGramLanguageModel below already
        implements this padding rule in get_possible_next_tokens - keep the two consistent, or
        the model will look up prefixes that you never counted.
        """
        if n <= 0:
            raise ValueError("test_invalid_n")

        counts = defaultdict(Counter)
        # counts[(word1, word2)][word3] = how many times word3 occured after (word1, word2)
        for line in lines:
            tokens = [self.UNK] * (n - 1) + line.split() + [self.EOS]
            for start in range(n - 1, len(tokens)):
                counts[tuple(tokens[start - n + 1:start])][tokens[start]] += 1

        return counts

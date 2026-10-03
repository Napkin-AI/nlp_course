"""Task 06: beam search generation."""

import torch
import numpy as np

BOS, EOS = ' ', '\n'

class BeamNode:
    def __init__(self, prefix, prob):
        self.prefix = prefix
        self.prob = prob

    def __hash__(self):
        return hash(self.prefix)

    def __eq__(self, other):
        return self.prefix == other.prefix

def generate_beamsearch(next_dist, prefix, beam_size=5, length=10):
    """Generate top hypotheses with beam search.

    Args:
        next_dist: callable that accepts a prefix string and returns
            ``{token: probability}``.
        prefix: initial text prefix.
        beam_size: number of hypotheses to keep.
        length: number of tokens to generate.

    Returns:
        A list of ``(text, log_probability)`` pairs sorted by score descending.
    """

    if length < 0 or beam_size < 1:
        raise ValueError("???")
    
    if length == 0:
        return [(prefix, 0.0)]
    
    def set_prefixes(prefix, prefix_prob):
        new_prefixes = list() 
        with torch.no_grad():
            token_probs = next_dist(prefix)
            tokens, probs = zip(*sorted(token_probs.items(), key=lambda x: -x[1]))
            for token, prob in zip(tokens[:beam_size], probs[:beam_size]):
                if prob > 0:
                    seq = prefix + ' ' + token if prefix else token
                    new_prefixes.append(BeamNode(seq, prefix_prob + np.log(prob)))
        return new_prefixes

    prefixes = set_prefixes(prefix, 0.0)
    for _ in range(length - 1):
        if all(node.prefix[-1] == EOS for node in prefixes):
            break
        
        new_prefixes = list()
        for node in prefixes:
            if node.prefix[-1] != EOS:
                new_prefixes += set_prefixes(node.prefix, node.prob)
            else:
                new_prefixes.append(node)
        prefixes = sorted(new_prefixes, key=lambda x: -x.prob)[:beam_size]

    return sorted([(x.prefix, x.prob) for x in prefixes], key=lambda x: -x[1])


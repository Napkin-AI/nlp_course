"""Task 04: corpus perplexity."""
import numpy as np

def perplexity(lm, lines, min_logprob=-50.0):
    """Compute perplexity of ``lm`` on a corpus of lines."""
    EOS = "_EOS_"
    if not len(lines):
        return 0.0
    
    score = 0.0
    N = 0
    for line in lines:
        tokens = line.split() + [EOS]
        context = ''

        for token in tokens:
            next_token_prob = lm.get_next_token_prob(context, token)
            logprob = max(np.log(next_token_prob), min_logprob)
            score -= logprob
            N += 1
            if token != EOS:
                context = token if context == "" else context + " " + token


    score /= N
    score = np.exp(score)
    return score

"""Task 03: temperature and nucleus sampling."""
import numpy as np
import random

def sample_with_temperature(dist, temperature=1.0, rng=None):
    """Sample one token from a probability distribution with temperature."""
    if rng is None:
        rng = random
    if len(dist) == 0:
        raise ValueError("Distribution is empty")
    if not hasattr(rng, "choices"):
        raise ValueError("Rng expected to have choise method")
    if temperature < 0:
        raise ValueError("To be sure RE wont happen")

    words, probs = list(dist.keys()), np.array(list(dist.values()), dtype=np.float64)
    probs /= probs.sum()
    if temperature == 1.0:
        return rng.choices(words, weights=probs)[0]
    
    if temperature == 0:
        return words[np.argmax(probs)]
    
    new_probs = np.log(probs)
    new_probs = np.exp((new_probs - np.max(new_probs)) / temperature)
    new_probs /= new_probs.sum()

    print(words, new_probs, rng.choices(words, weights=new_probs)[0]) 
    return rng.choices(words, weights=new_probs)[0]


def nucleus_filter(dist, nucleus=0.9):
    """Keep the smallest high-probability token set with cumulative mass >= nucleus."""
    if not (0 < nucleus <= 1):
        raise ValueError("To be sure RE wont happen")

    if nucleus == 1.0:
        return dist
    
    _dist = sorted(
        [(word, prob) for word, prob in dist.items()],
        key=lambda x: -x[1]
    )

    accumulate = 0
    new_dist = {}
    for word, prob in _dist:
        if accumulate < nucleus  or (len(new_dist) == 0):
            new_dist[word] = prob
            accumulate += prob
        else:
            break

    new_probs = np.array(list(new_dist.values()), dtype=np.float64)

    new_probs /= new_probs.sum()

    return {
        word: prob
        for word, prob in zip(new_dist.keys(), new_probs)
    }


def sample_nucleus(dist, nucleus=0.9, rng=None):
    """Sample one token after applying nucleus filtering."""
    if rng is None:
        rng = random
    if len(dist) == 0:
        raise ValueError("Distribution is empty")
    if not hasattr(rng, "choices"):
        raise ValueError("Rng expected to have choise method")
    if not (0 < nucleus <= 1):
        raise ValueError("To be sure RE wont happen")

    new_dst = nucleus_filter(dist, nucleus)
    words, probs = list(new_dst.keys()), np.array(list(new_dst.values()), dtype=np.float64)
    probs /= probs.sum()
    return rng.choices(words, weights=probs)[0]

# if __name__ == "__main__":
#     dist = {
#         "a": 0.7,
#         "b": 0.3,
#         "c": 0.0,
#     }

#     rng = None

#     samples = [
#         sample_with_temperature(dist, temperature=1.0, rng=rng)
#         for _ in range(50000)
#     ]

#     frequencies = {
#         word: samples.count(word) / len(samples)
#         for word in dist
#     }

#     print("temperature=1 frequencies:", frequencies)
    
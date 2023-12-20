import random
from collections import defaultdict


class Markov:
    EOW = None

    def __init__(self, *, tokenizer, order=1):
        self.tokenizer = tokenizer
        self.order = order
        self.tokens = defaultdict(lambda: defaultdict(lambda: 0))

    def add_token(self, word):
        tokens = self.tokenizer(word)

        fragments = []
        for i in range(self.order+1):
            fragments.append(tokens[i:])

        for causes_and_effect in zip(*fragments):
            cause, effect = causes_and_effect[:-1], causes_and_effect[-1]
            self.tokens[tuple(cause)][effect] += 1

    def generate(self, start=None, n_min=None, n_max=None):
        start = start or self.__random_token()

        word = [*start]
        while word[-1] is not None and (n_max is None or len(word) < n_max):
            next_cause = tuple(word[-self.order:])
            available_tokens = self.tokens[next_cause]
            if available_tokens:
                tokens, weights = zip(*available_tokens.items())
                word.append(random.choices(tokens, weights=weights)[0])
            else:
                break

            if n_min is not None and word[-1] is None and len(word) < n_min:
                word.pop()
                word.extend(self.__random_token())

        # Remove last element if it is EOW
        if word[-1] == Markov.EOW:
            word.pop()

        return word

    def __random_token(self):
        tokens, weights = zip(
            *[(k, len(v)) for k, v in self.tokens.items()]
        )
        return random.choices(tokens, weights=weights)[0]


m = Markov(tokenizer=lambda word: list(word), order=5)
with open("resources/star_names.txt", "r") as f:
    raw = f.read()
    planets = raw.split("\n")
    for p in planets:
        m.add_token(p)

for i in range(10):
    print(''.join(m.generate(n_min=3)).strip().capitalize())

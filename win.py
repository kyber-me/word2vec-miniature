import numpy as np
from numpy.typing import NDArray


CHUNK_SIZE = 5
CONTEXT_SIZE = CHUNK_SIZE - 1


class WeightMatrix:
    corpora: NDArray
    vocabulary: NDArray[np.str_] | None = None
    embeddings: NDArray[np.float64] | None = None

    def __init__(self, path: str = "corpora.csv") -> None:
        try:
            self.corpora = np.loadtxt(path, dtype=str, delimiter=";", skiprows=1)

            self.vocabulary = np.load("vocabulary.npy", "r")
            self.embeddings = np.load("embeddings.npy", "r")

        except Exception as e:
            print(e)

    def get_phrases(self) -> NDArray:
        return self.corpora[:, -1] if self.corpora is not None else np.array([])

    # --- Public API ---

    def build(self) -> NDArray:
        return (
            self.embeddings
            if self.embeddings
            else self._build_vocabulary()._build_matrix()
        )

    # --- Private functions ---

    def _build_vocabulary(self) -> "WeightMatrix":

        phrases = self.get_phrases()
        vocabulary = np.array(
            sorted(
                list(
                    set(
                        [word for phrase in phrases for word in phrase.split()],
                    )
                ),
                key=lambda x: x,
            )
        )

        np.save("vocabulary.npy", vocabulary)
        self.vocabulary = vocabulary

        return self

    def _build_matrix(self) -> NDArray:

        if self.vocabulary is None or not len(self.vocabulary):
            raise Exception("No vocabulary is found")

        rng = np.random.default_rng()
        out = np.array([rng.standard_normal(3) for _ in self.vocabulary])
        self.embeddings = out
        np.save("embeddings.npy", out)

        return out


if __name__ == "__main__":
    wm = WeightMatrix()
    embeddings = wm.embeddings if wm.embeddings is not None else wm.build()
    # if not wm.embeddings:
    #     embeddings = wm.embeddings if wm.embeddings is not None else wm.build()

    if wm.vocabulary is not None and wm.embeddings is not None:
        # Plotar o estado inicial dos pontos no espaço vetorial semântico
        # ...
        for e, w in zip(wm.embeddings, wm.vocabulary):
            print("Plot the points")
            print(w, " - ", e)

    phrases = wm.get_phrases()
    chunks = []
    for p in phrases:
        # [0, +1, +2, +3, +4] - first word
        # [-1, 0, +1, +2, +3] - second word

        # [-2, -1, 0, +1, +2] - middle of sentence (several times) -> zero word index 2 (TWO)

        # [-3, -2, -1, 0, +1] - second last word
        # [-4, -3, -2, -1, 0] -> last word
        # [embedding_0, embedding_1, embedding_2, ..., embedding_n] = [0, 0, 1, ..., 0] -> label of the data where one represents the expected value for that training set
        p_size = len(p)

        # chunk = ['', '', '', '', ''] - is there another way to be more straightforward with minimal code?
        chunk = np.zeros(5)
        for zero_word_index, w in enumerate(
            p
        ):  # zero_word_index overcomes 5 and it has to be circular (?)
            chunk[zero_word_index] = w
            # zero word index equals 0 (ZERO) ->
            # in a space of 5 words

            pass

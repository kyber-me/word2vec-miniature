import numpy as np
from numpy.typing import NDArray


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

    def _get_phrases(self) -> NDArray:
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

        phrases = self._get_phrases()
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

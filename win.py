import numpy as np
from numpy.typing import NDArray


class Win:
    corpora: NDArray
    vocabulary: NDArray | None

    def __init__(self, path: str = "corpora.csv") -> None:
        try:
            self.corpora = np.loadtxt(path, dtype=str, delimiter=";", skiprows=1)
        except Exception as e:
            print(e)

            raise Exception("System failure")

    # --- Public API ---

    def build(self) -> NDArray:
        return self._build_vocabulary()._build_matrix()

    # --- Private functions ---

    def _build_vocabulary(self) -> "Win":
        phrases = self.corpora[:, -1]
        # print("phrases: ", phrases)

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
        # print("vocabulary: ", vocabulary)

        np.save("vocabulary.npy", vocabulary)
        self.vocabulary = vocabulary

        return self

    def _build_matrix(self) -> NDArray:

        if self.vocabulary is None or not len(self.vocabulary):
            raise Exception("No vocabulary is found")

        rng = np.random.default_rng()
        out = np.array([rng.standard_normal(3) for _ in self.vocabulary])

        print(f"out -> {out}")

        self.embeddings = out

        np.save("embeddings.npy", out)

        return out


if __name__ == "__main__":
    win = Win()
    win.build()

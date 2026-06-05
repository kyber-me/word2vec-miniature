import numpy as np
from numpy.typing import NDArray


class Win:
    corpora: NDArray

    def __init__(self, path: str = "corpora.csv") -> None:
        try:
            # self.corpora = np.genfromtxt(path, delimiter=";")
            self.corpora = np.loadtxt(path, dtype=str, delimiter=";", skiprows=1)
        except Exception as e:
            print(e)

    def build_matrix(self):
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


if __name__ == "__main__":
    win = Win()
    win.build_matrix()

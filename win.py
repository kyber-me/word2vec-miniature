import numpy as np
from numpy.typing import NDArray


CHUNK_SIZE = 5
CONTEXT_SIZE = CHUNK_SIZE - 1


class WeightMatrix:
    _corpora: NDArray
    _dataset: NDArray[np.str_] | None = None
    _embeddings: NDArray[np.float64] | None = None

    word_to_index: dict

    paths: dict = {
        "corpora": "corpora.csv",
        "dataset": "dataset.npy",
        "embeddings": "embeddings.npy",
    }

    def __init__(self) -> None:
        self._corpora = self._load_corpora()
        self._vocabulary = self._load_vocabulary()
        self._embeddings = self._load_embeddings()

        self.word_to_index = {}

    def _load_corpora(self) -> NDArray:
        try:
            self._corpora = np.loadtxt(
                self.paths["corpora"], dtype=str, delimiter=";", skiprows=1
            )

            return self._corpora
        except Exception as e:
            print(e)
            raise Exception("corpora not found")

    def _load_vocabulary(self) -> NDArray | None:
        try:
            dataset = np.load(self.paths["dataset"], "r")
        except Exception as e:
            print(e, " NOT CREATED YET")
            return None

        return dataset

    def _load_embeddings(self) -> NDArray | None:
        try:
            embeddings = np.load(self.paths["embeddings"], "r")
        except Exception as e:
            print(e, " – NOT CREATED YET")
            return None

        return embeddings

    @property
    def corpora(self) -> NDArray:
        return self._corpora[:, -1]

    @property
    def vocabulary(self) -> NDArray:
        if self._vocabulary is None:
            raise Exception("You must call `build_and_retrieve` before calling dataset")

        return self._vocabulary

    @property
    def embeddings(self) -> NDArray:
        if self._embeddings is None:
            raise Exception()

        return self._embeddings

    # --- Public API --- ?

    def build_and_retrieve(self):
        self._build_dataset()._build_matrix()

    # --- Private functions ---

    def _build_dataset(self) -> "WeightMatrix":

        sentences = self.corpora
        self._vocabulary = np.array(
            sorted(
                list(
                    set(
                        [word for phrase in sentences for word in phrase.split()],
                    )
                ),
                key=lambda x: x,
            )
        )

        np.save("vocabulary.npy", self._vocabulary)
        return self

    def _build_matrix(self):
        if self._vocabulary is None:
            raise Exception()

        rng = np.random.default_rng()
        embeddings = []
        for i, w in enumerate(self._vocabulary):
            # Um embedding está sendo criado randomicamente para a palavra w numa posição i específica no array
            # de embeddings AND
            # criamos uma tabela para procurar o índice dessa palavra em embeddings (palavra -> index)
            embeddings.append(rng.standard_normal(3))
            self.word_to_index[w] = i

        # self._embeddings = np.array([rng.standard_normal(3) for _ in self._vocabulary])
        self._embeddings = np.array(embeddings)
        np.save("embeddings.npy", self._embeddings)


if __name__ == "__main__":
    context_filter = [
        [1, 2, 3, 4],
        [-1, 1, 2, 3],
        [-2, -1, 1, 2],
        [-3, -2, -1, 1],
        [-4, -3, -2, -1],
    ]

    wm = WeightMatrix()
    wm.build_and_retrieve()
    # If build and retrieve is not called before, calling vocabulary raises an exception
    vocabulary = wm.vocabulary
    embeddings = wm.embeddings

    # Plotar o estado inicial dos pontos no espaço vetorial semântico
    # ...
    for e, w in zip(embeddings, vocabulary):
        print("Plot the points")
        print(w, " - ", e)

    corpora = wm.corpora
    dataset = np.array([])
    for sentence in corpora:
        # s_size = len(sentence)
        # print("len of the sentence: ", s_size)

        # chunk = np.zeros(5)
        for target_word_idx, w in enumerate(sentence):
            map_filter = context_filter[
                target_word_idx
                if target_word_idx in [0, 1, len(sentence) - 2, len(sentence) - 1]
                else 2
            ]

            # Preciso recuperar os embeddings para cada palavra
            # Melhor forma de recuperar embeddings de palavras é através de alguma estrutura de dados
            # que me permita chavear uma palavra para um índice (word to index)

            # O quê o sistema já tem?
            # 1 - Um array numpy representando o vocabulário
            # 2 - Um array numpy representando os embeddings
            # - As 2 estruturas seguem a mesma ordem mas eu precisaria achar o index da palavra com algo como `indexof`
            # pra usar como índice na estrutura de embeddings
            target = w
            context = [
                wm.word_to_index[sentence[target_word_idx + i]] for i in map_filter
            ]

            print("context: ", context)

        # [embedding_0, embedding_1, embedding_2, ..., embedding_n] = [0, 0, 1, ..., 0] -> label of the data where one represents the expected value for that training set

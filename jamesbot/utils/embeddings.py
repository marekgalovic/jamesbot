class EmbeddingHandler(object):

    def __init__(self, dictionary, missing_id = 2):
        self._dictionary = dictionary
        self._index = {val: key for (key, val) in dictionary.items()}
        self._missing_id = int(missing_id)

    def __len__(self):
        return len(self._dictionary)

    def get_embedding(self, token):
        return self._dictionary.get(token, self._missing_id)

    def get_token(self, embedding_id):
        return self._index[embedding_id]

    def embeddings(self, tokens):
        return list(map(self.get_embedding, tokens))

    def tokens(self, embeddings):
        return list(map(self.get_token, embeddings))

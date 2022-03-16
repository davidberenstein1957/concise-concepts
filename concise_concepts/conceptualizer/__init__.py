

import itertools

from gensim.models.keyedvectors import KeyedVectors
from spacy import util
from spacy.tokens import Doc


class ConceptualSpacy:
    def __init__(self, nlp, name, data, topn=[]):
        self.data = data
        self.name = name
        self.nlp = nlp
        self.topn = topn
        self.run()

    def run(self):
        self.determine_topn()
        self.set_gensim_model()
        self.expand_concepts()
        self.resolve_overlapping_concepts()
        self.lemmatize_concepts()
        self.create_conceptual_patterns()
        del self.kv

    def determine_topn(self):
        self.topn_dict = {}
        if not self.topn:
            for key in self.data:
                self.topn_dict[key] = 150
        else:
            try:
                num_classes = len(list(self.data.keys()))
                assert len(self.topn) == num_classes
                for key, n in zip(self.data, self.topn):
                    self.topn_dict[key] = n
            except Exception as _:
                raise Exception(
                    f'Provide a topn integer for each of the {num_classes} classes.')

    def set_gensim_model(self):
        wordList = []
        vectorList = []

        try:
            assert len(self.nlp.vocab.vectors)
        except Exception as _:
            raise Exception(
                'Choose a model with internal embeddings i.e. md or lg.')

        for key, vector in self.nlp.vocab.vectors.items():
            wordList.append(self.nlp.vocab.strings[key])
            vectorList.append(vector)

        self.kv = KeyedVectors(self.nlp.vocab.vectors_length)

        self.kv.add_vectors(wordList, vectorList)

    def expand_concepts(self):
        for key in self.data:
            remaining_keys = [
                rem_key for rem_key in self.data.keys() if rem_key != key]
            remaining_values = [self.data[rem_key]
                                for rem_key in remaining_keys]
            remaining_values = list(
                itertools.chain.from_iterable(remaining_values))
            similar = self.kv.most_similar(
                positive=self.data[key] + [key],
                # negative=remaining_values,
                topn=self.topn_dict[key]
            )
            similar = [sim_pair[0] for sim_pair in similar]
            self.data[key] += similar
            self.data[key] = list(set([word.lower()
                                  for word in self.data[key]]))

    def resolve_overlapping_concepts(self):
        centroids = {}
        for key in self.data:
            if key not in self.kv:
                words = self.data[key]
                while len(words) != 1:
                    words.remove(self.kv.doesnt_match(words))
                centroids[key] = words[0]
            else:
                centroids[key] = key

        for key in self.data:
            self.data[key] = [
                word for word in self.data[key]
                if centroids[key] == self.kv.most_similar_to_given(word, list(centroids.values()))
            ]

    def lemmatize_concepts(self):
        for key in self.data:
            self.data[key] = list(
                set([doc[0].lemma_ for doc in self.nlp.pipe(self.data[key])]))

    def create_conceptual_patterns(self):
        patterns = []
        for key in self.data:
            for word in self.data[key]:
                if word != key:
                    default_pattern = {
                        "lemma": {"regex": r"(?i)"+word}, 'POS': {'NOT_IN': ['VERB']}}

                    patterns.append(
                        {
                            'label': key.upper(),
                            'pattern': [{'DEP': {'IN': ['amod', 'compound']}, 'OP': '?'},
                                        default_pattern,
                                        {'DEP': {'IN': ['amod', 'compound']}, 'OP': '?'}],
                            'id': key
                        }
                    )
        self.ruler = self.nlp.add_pipe("entity_ruler", config={
            "overwrite_ents": True
        })
        self.ruler.add_patterns(patterns)

    def __call__(self, doc: Doc):
        return doc

    def pipe(self, stream, batch_size=128):
        for docs in util.minibatch(stream, size=batch_size):
            for doc in docs:
                yield doc

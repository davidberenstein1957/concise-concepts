import itertools
from copy import deepcopy

import gensim.downloader
from gensim.models import FastText, Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from spacy import Language, util
from spacy.tokens import Doc, Span


class ConceptualSpacy:
    def __init__(self, nlp: Language, name: str, data: dict, topn: list = None, model_path=None, ent_score=False):
        if Span.has_extension("ent_score"):
            Span.remove_extension("ent_score")
        if ent_score:
            Span.set_extension("ent_score", default=None)
        self.ent_score = ent_score
        self.orignal_words = [j for i in data.values() for j in i]
        self.original_data = deepcopy(data)
        self.data = data
        self.name = name
        self.nlp = nlp
        self.topn = topn
        self.model_path = model_path
        self.run()
        self.data_upper = {k.upper(): v for k, v in data.items()}

    def run(self):
        self.determine_topn()
        self.set_gensim_model()
        self.expand_concepts()
        # settle words around overlapping concepts
        for _ in range(5):
            self.expand_concepts()
            self.infer_original_data()
            self.resolve_overlapping_concepts()
        self.infer_original_data()
        self.create_conceptual_patterns()

        if not self.ent_score:
            del self.kv

    def determine_topn(self):
        self.topn_dict = {}
        if not self.topn:
            for key in self.data:
                self.topn_dict[key] = 100
        else:
            num_classes = len(list(self.data.keys()))
            assert len(self.topn) == num_classes, f"Provide a topn integer for each of the {num_classes} classes."
            for key, n in zip(self.data, self.topn):
                self.topn_dict[key] = n

    def set_gensim_model(self):
        if self.model_path:
            available_models = gensim.downloader.info()["models"]
            if self.model_path in available_models:
                self.kv = gensim.downloader.load(self.model_path)
            else:
                try:
                    self.kv = FastText.load(self.model_path)
                except Exception as e1:
                    try:
                        self.kv = Word2Vec.load(self.model_path)
                    except Exception as e2:
                        try:
                            self.kv = KeyedVectors.load(self.model_path)
                        except Exception as e3:
                            raise Exception(
                                f"Not a valid gensim model. FastText, Word2Vec, KeyedVectors.\n {e1}\n {e2}\n {e3}"
                            )

        else:
            wordList = []
            vectorList = []

            assert len(self.nlp.vocab.vectors), "Choose a model with internal embeddings i.e. md or lg."

            for key, vector in self.nlp.vocab.vectors.items():
                wordList.append(self.nlp.vocab.strings[key])
                vectorList.append(vector)

            self.kv = KeyedVectors(self.nlp.vocab.vectors_length)

            self.kv.add_vectors(wordList, vectorList)

    def expand_concepts(self):
        for key in self.data:
            remaining_keys = [rem_key for rem_key in self.data.keys() if rem_key != key]
            remaining_values = [self.data[rem_key] for rem_key in remaining_keys]
            remaining_values = list(itertools.chain.from_iterable(remaining_values))
            similar = self.kv.most_similar(
                positive=self.data[key] + [key],
                topn=self.topn_dict[key],
            )
            similar = [sim_pair[0] for sim_pair in similar]
            self.data[key] += similar
            self.data[key] = list(set([word.lower() for word in self.data[key]]))

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

        for key_x in self.data:
            for key_y in self.data:
                if key_x != key_y:
                    self.data[key_x] = [word for word in self.data[key_x] if word not in self.original_data[key_y]]

        for key in self.data:
            self.data[key] = [
                word
                for word in self.data[key]
                if centroids[key] == self.kv.most_similar_to_given(word, list(centroids.values()))
            ]

        self.centroids = centroids

    def infer_original_data(self):
        data = deepcopy(self.original_data)
        for key in self.data:
            self.data[key] += data[key]
            self.data[key] = list(set(self.data[key]))

        for key_x in self.data:
            for key_y in self.data:
                if key_x != key_y:
                    self.data[key_x] = [word for word in self.data[key_x] if word not in self.original_data[key_y]]

    def lemmatize_concepts(self):
        for key in self.data:
            self.data[key] = list(set([doc[0].lemma_ for doc in self.nlp.pipe(self.data[key])]))

    def create_conceptual_patterns(self):
        patterns = []
        for key in self.data:
            for word in self.data[key]:
                if word != key:
                    individual_pattern = {
                        "lemma": {"regex": r"(?i)" + word},
                        "POS": {"NOT_IN": ["VERB"]},
                        "DEP": {"NOT_IN": ["nsubjpass"]},
                    }

                    patterns.append(
                        {
                            "label": key.upper(),
                            "pattern": [
                                individual_pattern,
                            ],
                            "id": f"{key}_individual",
                        }
                    )

                    default_pattern = {
                        "lemma": {"regex": r"(?i)" + word},
                        "POS": {"NOT_IN": ["VERB"]},
                        "DEP": {"NOT_IN": ["nsubjpass", "compound"]},
                    }

                    patterns.append(
                        {
                            "label": key.upper(),
                            "pattern": [
                                {"DEP": {"IN": ["amod", "compound"]}, "OP": "?"},
                                default_pattern,
                                {"DEP": {"IN": ["amod", "compound"]}, "OP": "?"},
                            ],
                            "id": f"{key}_compound",
                        }
                    )
        self.ruler = self.nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
        self.ruler.add_patterns(patterns)

    def __call__(self, doc: Doc):
        if self.ent_score:
            doc = self.assign_score_to_entities(doc)
        return doc

    def pipe(self, stream, batch_size=128):
        for docs in util.minibatch(stream, size=batch_size):
            for doc in docs:
                if self.ent_score:
                    doc = self.assign_score_to_entities(doc)
                yield doc

    def assign_score_to_entities(self, doc: Doc):
        ents = doc.ents
        for ent in ents:
            if ent.label_ in self.data_upper:
                entity = [part for part in ent.text.split() if part in self.kv]
                concept = [word for word in self.data_upper[ent.label_] if word in self.kv]
                if entity and concept:
                    ent._.ent_score = self.kv.n_similarity(entity, concept)
                else:
                    ent._.ent_score = 1
            else:
                ent._.ent_score = 1
        doc.ents = ents
        return doc

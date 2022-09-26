# -*- coding: utf-8 -*-
import itertools
import json
import logging
import re
from copy import deepcopy

import gensim.downloader
from gensim.models import FastText, Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from spacy import Language, util
from spacy.tokens import Doc, Span

logger = logging.getLogger(__name__)


class Conceptualizer:
    def __init__(
        self,
        nlp: Language,
        name: str,
        data: dict,
        topn: list = None,
        model_path=None,
        word_delimiter="_",
        ent_score=False,
        exclude_pos=[],
        exclude_dep=[],
        include_compound_words=False,
        case_sensitive=False,
    ):
        """
        The function takes in a dictionary of words and their synonyms, and then creates a new dictionary of words and
        their synonyms, but with the words in the new dictionary all in uppercase

        :param nlp: The spaCy model to use.
        :type nlp: Language
        :param name: The name of the entity.
        :type name: str
        :param data: A dictionary of the words you want to match. The keys are the classes you want to match,
            and the values are the words you want to expand over.
        :type data: dict
        :param topn(): The number of words to be returned for each class.
        :type topn: list
        :param model_path: The path to the model you want to use. If you don't have a model, you can use the spaCy one.
        :param word_delimiter: The delimiter used to separate words in model the dictionary, defaults to _ (optional)
        :param ent_score: If True, the extension "ent_score" will be added to the Span object. This will be the score of
            the entity, defaults to False (optional)
        :param exclude_pos: A list of POS tags to exclude from the rule based match
        :param exclude_dep: list of dependencies to exclude from the rule based match
        :param include_compound_words: If True, it will include compound words in the entity. For example,
            if the entity is "New York", it will also include "New York City" as an entity, defaults to False (optional)
        :param case_sensitive: Whether to match the case of the words in the text, defaults to False (optional)
        """
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
        self.match_rule = {}
        if exclude_pos:
            self.match_rule["POS"] = {"NOT_IN": exclude_pos}
        if exclude_dep:
            self.match_rule["DEP"] = {"NOT_IN": exclude_dep}

        self.include_compound_words = include_compound_words
        self.case_sensitive = case_sensitive
        self.word_delimiter = word_delimiter
        if "lemmatizer" not in self.nlp.component_names:
            logger.warning(
                "No lemmatizer found in spacy pipeline. Consider adding it for matching"
                " on LEMMA instead of exact text."
            )
            self.match_key = "ORTH"
        else:
            self.match_key = "LEMMA"
        self.run()
        self.data_upper = {k.upper(): v for k, v in data.items()}

    def run(self):
        self.determine_topn()
        self.set_gensim_model()
        self.verify_data()
        self.expand_concepts()
        self.verify_data(verbose=False)
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
        """
        If the user doesn't specify a topn value for each class,
        then the topn value for each class is set to 100
        """
        self.topn_dict = {}
        if not self.topn:
            for key in self.data:
                self.topn_dict[key] = 100
        else:
            num_classes = len(list(self.data.keys()))
            assert (
                len(self.topn) == num_classes
            ), f"Provide a topn integer for each of the {num_classes} classes."
            for key, n in zip(self.data, self.topn):
                self.topn_dict[key] = n

    def set_gensim_model(self):
        """
        If the model_path is not None, then we try to load the model from the path.
        If it's not a valid path, then we raise an exception.
        If the model_path is None, then we load the model from the internal embeddings of the spacy model
        """
        if self.model_path:
            available_models = gensim.downloader.info()["models"]
            if self.model_path in available_models:
                self.kv = gensim.downloader.load(self.model_path)
            else:
                try:
                    self.kv = FastText.load(self.model_path).wv
                except Exception as e1:
                    try:
                        self.kv = Word2Vec.load(self.model_path).wv
                    except Exception as e2:
                        try:
                            self.kv = KeyedVectors.load(self.model_path)
                        except Exception as e3:
                            raise Exception(
                                "Not a valid gensim model. FastText, Word2Vec,"
                                f" KeyedVectors.\n {e1}\n {e2}\n {e3}"
                            )
        else:
            wordList = []
            vectorList = []

            assert len(
                self.nlp.vocab.vectors
            ), "Choose a model with internal embeddings i.e. md or lg."

            for key, vector in self.nlp.vocab.vectors.items():
                wordList.append(self.nlp.vocab.strings[key])
                vectorList.append(vector)

            self.kv = KeyedVectors(self.nlp.vocab.vectors_length)

            self.kv.add_vectors(wordList, vectorList)

    def verify_data(self, verbose: bool = True):
        """
        It takes a dictionary of lists of words, and returns a dictionary of lists of words,
        where each word in the list is present in the word2vec model
        """
        verified_data = {}
        for key, value in self.data.items():
            verified_values = []
            if not self.check_presence_vocab(key):
                if verbose:
                    logger.warning(f"key {key} not present in word2vec model")
            for word in value:
                if self.check_presence_vocab(word):
                    verified_values.append(self.check_presence_vocab(word))
                else:
                    if verbose:
                        logger.warning(
                            f"word {word} from key {key} not present in word2vec model"
                        )
            verified_data[key] = verified_values
            assert len(
                verified_values
            ), f"None of the entries for key {key} are present in the word2vec model"
        self.data = deepcopy(verified_data)
        self.original_data = deepcopy(self.data)

    def expand_concepts(self):
        """
        For each key in the data dictionary, find the topn most similar words to the key and the values in the data
        dictionary, and add those words to the values in the data dictionary
        """

        for key in self.data:
            remaining_keys = [rem_key for rem_key in self.data.keys() if rem_key != key]
            remaining_values = [self.data[rem_key] for rem_key in remaining_keys]
            remaining_values = list(itertools.chain.from_iterable(remaining_values))
            if self.check_presence_vocab(key):
                key_list = [self.check_presence_vocab(key)]
            else:
                key_list = []
            similar = self.kv.most_similar(
                positive=self.data[key] + key_list,
                topn=self.topn_dict[key],
            )
            similar = [sim_pair[0] for sim_pair in similar]
            self.data[key] += similar
            self.data[key] = list(set([word.lower() for word in self.data[key]]))

    def resolve_overlapping_concepts(self):
        """
        It removes words from the data that are in other concepts, and then removes words that are not closest to the
        centroid of the concept
        """
        centroids = {}
        for key in self.data:
            if not self.check_presence_vocab(key):
                words = self.data[key]
                while len(words) != 1:
                    words.remove(self.kv.doesnt_match(words))
                centroids[key] = words[0]
            else:
                centroids[key] = key

        for key_x in self.data:
            for key_y in self.data:
                if key_x != key_y:
                    self.data[key_x] = [
                        word
                        for word in self.data[key_x]
                        if word not in self.original_data[key_y]
                    ]

        for key in self.data:
            self.data[key] = [
                word
                for word in self.data[key]
                if centroids[key]
                == self.kv.most_similar_to_given(word, list(centroids.values()))
            ]

        self.centroids = centroids

    def infer_original_data(self):
        """
        It takes the original data and adds the new data to it, then removes the new data from the original data.
        """
        data = deepcopy(self.original_data)
        for key in self.data:
            self.data[key] += data[key]
            self.data[key] = list(set(self.data[key]))

        for key_x in self.data:
            for key_y in self.data:
                if key_x != key_y:
                    self.data[key_x] = [
                        word
                        for word in self.data[key_x]
                        if word not in self.original_data[key_y]
                    ]

        self.verify_data(verbose=False)

    def lemmatize_concepts(self):
        """
        For each key in the data dictionary,
        the function takes the list of concepts associated with that key, and lemmatizes
        each concept.
        """
        for key in self.data:
            self.data[key] = list(
                set([doc[0].lemma_ for doc in self.nlp.pipe(self.data[key])])
            )

    def create_conceptual_patterns(self):
        """
        For each key in the data dictionary,
        create a pattern for each word in the list of words associated with that key.


        The pattern is a dictionary with three keys:

        1. "lemma"
        2. "POS"
        3. "DEP"

        The value for each key is another dictionary with one key and one value.

        The key is either "regex" or "NOT_IN" or "IN".

        The value is either a regular expression or a list of strings.

        The regular expression is the word associated with the key in the data dictionary.

        The list of strings is either ["VERB"] or ["nsubjpass"] or ["amod", "compound"].

        The regular expression is case insensitive.

        The pattern is
        """
        patterns = []

        def add_patterns(input_dict):
            for key in input_dict:
                if self.match_key == "LEMMA":
                    words = [
                        "".join(
                            [
                                token.lemma_ if token.lemma_ else token.text
                                for token in doc
                            ]
                        )
                        for doc in self.nlp.pipe(input_dict[key])
                    ]
                else:
                    words = input_dict[key]
                for word in words:
                    if word != key:
                        specific_copy = deepcopy(self.match_rule)
                        word_parts = re.split(f"[{self.word_delimiter}]+", word)
                        if len(word_parts) > 1:
                            operators = [" ", "-"]
                        else:
                            operators = [""]

                        for op in operators:
                            if self.case_sensitive:
                                specific_copy[self.match_key] = "{op}".join(word_parts)
                            else:
                                specific_copy[self.match_key] = {
                                    "regex": r"(?i)"
                                    + re.escape(f"{op}".join(word_parts))
                                }

                            patterns.append(
                                {
                                    "label": key.upper(),
                                    "pattern": [
                                        specific_copy,
                                    ],
                                    "id": f"{word}_{op}_individual",
                                }
                            )

                            if self.include_compound_words:
                                compound_rule = {
                                    "DEP": {"IN": ["amod", "compound"]},
                                    "OP": "?",
                                }
                                patterns.append(
                                    {
                                        "label": key.upper(),
                                        "pattern": [
                                            compound_rule,
                                            specific_copy,
                                            compound_rule,
                                        ],
                                        "id": f"{word}_{op}_compound",
                                    }
                                )

        add_patterns(self.data)
        add_patterns(self.original_data)
        with open("matching_patterns.json", "w") as f:
            json.dump(patterns, f)
        self.ruler = self.nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
        self.ruler.add_patterns(patterns)

    def __call__(self, doc: Doc):
        """
        It takes a doc object and assigns a score to each entity in the doc object

        :param doc: Doc
        :type doc: Doc
        """
        if self.ent_score:
            doc = self.assign_score_to_entities(doc)
        return doc

    def pipe(self, stream, batch_size=128):
        """
        It takes a stream of documents, and for each document,
        it assigns a score to each entity in the document

        :param stream: a generator of documents
        :param batch_size: The number of documents to be processed at a time, defaults to 128 (optional)
        """
        for docs in util.minibatch(stream, size=batch_size):
            for doc in docs:
                if self.ent_score:
                    doc = self.assign_score_to_entities(doc)
                yield doc

    def assign_score_to_entities(self, doc: Doc):
        """
        The function takes a spaCy document as input and assigns a score to each entity in the document. The score is
        calculated using the word embeddings of the entity and the concept.
        The score is assigned to the entity using the
        `._.ent_score` attribute

        :param doc: Doc
        :type doc: Doc
        :return: The doc object with the entities and their scores.
        """
        ents = doc.ents
        for ent in ents:
            if ent.label_ in self.data_upper:
                entity = [
                    self.check_presence_vocab(part)
                    for part in ent.text.split()
                    if self.check_presence_vocab(part)
                ]
                concept = [
                    self.check_presence_vocab(word)
                    for word in self.data_upper[ent.label_]
                    if word in self.check_presence_vocab(word)
                ]
                if entity and concept:
                    ent._.ent_score = self.kv.n_similarity(entity, concept)
                else:
                    ent._.ent_score = 0
                    logger.warning(
                        f"Entity {entity} not found in model. Setting score to 0."
                    )
            else:
                ent._.ent_score = 0
                logger.warning(
                    f"Entity {entity} not found in model. Setting score to 0."
                )
        doc.ents = ents
        return doc

    def check_presence_vocab(self, word):
        for op in [" ", "-"]:
            if word.replace(op, self.word_delimiter) in self.kv:
                return word.replace(op, self.word_delimiter)

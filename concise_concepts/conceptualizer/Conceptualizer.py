# -*- coding: utf-8 -*-
import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import List

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
        model_path: str = None,
        word_delimiter: str = "_",
        ent_score: bool = False,
        exclude_pos: list = None,
        exclude_dep: list = None,
        include_compound_words: bool = False,
        case_sensitive: bool = False,
        json_path: str = "./matching_patterns.json",
        verbose: bool = True,
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
        self.verbose = verbose
        self.log_cache = {"key": list(), "word": list(), "word_key": list()}
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
        if exclude_pos is None:
            exclude_pos = [
                "VERB",
                "AUX",
                "ADP",
                "DET",
                "CCONJ",
                "PUNCT",
                "ADV",
                "ADJ",
                "PART",
                "PRON",
            ]
        if exclude_pos:
            self.match_rule["POS"] = {"NOT_IN": exclude_pos}
        if exclude_dep is None:
            exclude_dep = []
        if exclude_dep:
            self.match_rule["DEP"] = {"NOT_IN": exclude_dep}
        self.json_path = json_path
        self.check_validity_path()
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

    def run(self) -> None:
        self.check_validity_path()
        self.determine_topn()
        self.set_gensim_model()
        self.verify_data(self.verbose)
        self.expand_concepts()
        # settle words around overlapping concepts
        for _ in range(5):
            self.expand_concepts()
            self.infer_original_data()
            self.resolve_overlapping_concepts()
        self.infer_original_data()
        self.create_conceptual_patterns()
        self.set_concept_dict()

        if not self.ent_score:
            del self.kv

        self.data_upper = {k.upper(): v for k, v in self.data.items()}

    def check_validity_path(self) -> None:
        """
        If the path is a file, create the parent directory if it doesn't exist. If the path is a directory, create the
        directory and set the path to the default file name
        """
        if self.json_path:
            if Path(self.json_path).suffix:
                Path(self.json_path).parents[0].mkdir(parents=True, exist_ok=True)
            else:
                Path(self.json_path).mkdir(parents=True, exist_ok=True)
                old_path = str(self.json_path)
                self.json_path = Path(self.json_path) / "matching_patterns.json"
                logger.warning(
                    f"Path ´{old_path} is a directory, not a file. Setting"
                    f" ´json_path´to {self.json_path}"
                )

    def determine_topn(self) -> None:
        """
        If the user doesn't specify a topn value for each class,
        then the topn value for each class is set to 100
        """
        if not self.topn:
            self.topn_dict = {key: 100 for key in self.data}
        else:
            num_classes = len(self.data)
            assert (
                len(self.topn) == num_classes
            ), f"Provide a topn integer for each of the {num_classes} classes."
            self.topn_dict = dict(zip(self.data, self.topn))

    def set_gensim_model(self) -> None:
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
            ), "Choose a spaCy model with internal embeddings, e.g. md or lg."

            for key, vector in self.nlp.vocab.vectors.items():
                wordList.append(self.nlp.vocab.strings[key])
                vectorList.append(vector)

            self.kv = KeyedVectors(self.nlp.vocab.vectors_length)

            self.kv.add_vectors(wordList, vectorList)

    def verify_data(self, verbose: bool = True) -> None:
        """
        It takes a dictionary of lists of words, and returns a dictionary of lists of words,
        where each word in the list is present in the word2vec model
        """
        verified_data: dict[str, list[str]] = dict()
        for key, value in self.data.items():
            verified_values = []
            present_key = self._check_presence_vocab(key)
            if not present_key and verbose and key not in self.log_cache["key"]:
                logger.warning(f"key ´{key}´ not present in vector model")
                self.log_cache["key"].append(key)
            for word in value:
                present_word = self._check_presence_vocab(word)
                if present_word:
                    verified_values.append(present_word)
                elif verbose and word not in self.log_cache["word"]:
                    logger.warning(
                        f"word ´{word}´ from key ´{key}´ not present in vector model"
                    )
                    self.log_cache["word"].append(word)
            verified_data[key] = verified_values
            if not len(verified_values):
                msg = (
                    f"None of the entries for key {key} are present in the vector"
                    " model. "
                )
                if present_key:
                    logger.warning(
                        msg + f"Using {present_key} as word to expand over instead."
                    )
                    verified_data[key] = present_key
                else:
                    raise Exception(msg)
        self.data = deepcopy(verified_data)

    def expand_concepts(self) -> None:
        """
        For each key in the data dictionary, find the topn most similar words to the key and the values in the data
        dictionary, and add those words to the values in the data dictionary
        """

        self.original_data = deepcopy(self.data)

        for key in self.data:
            present_key = self._check_presence_vocab(key)
            if present_key:
                key_list = [present_key]
            else:
                key_list = []
            similar = self.kv.most_similar(
                positive=self.data[key] + key_list,
                topn=self.topn_dict[key],
            )
            self.data[key] = list(
                {self._check_presence_vocab(word) for word, _ratio in similar}
            )

    def resolve_overlapping_concepts(self) -> None:
        """
        It removes words from the data that are in other concepts, and then removes words that are not closest to the
        centroid of the concept
        """
        for key in self.data:
            self.data[key] = [
                word
                for word in self.data[key]
                if key == self.kv.most_similar_to_given(word, list(self.data.keys()))
            ]

    def infer_original_data(self) -> None:
        """
        It takes the original data and adds the new data to it, then removes the new data from the original data.
        """
        for key in self.data:
            self.data[key] = list(set(self.data[key] + self.original_data[key]))

        for key_x in self.data:
            for key_y in self.data:
                if key_x != key_y:
                    self.data[key_x] = [
                        word
                        for word in self.data[key_x]
                        if word not in self.original_data[key_y]
                    ]

    def lemmatize_concepts(self) -> None:
        """
        For each key in the data dictionary,
        the function takes the list of concepts associated with that key, and lemmatizes
        each concept.
        """
        for key in self.data:
            self.data[key] = list(
                set([doc[0].lemma_ for doc in self.nlp.pipe(self.data[key])])
            )

    def create_conceptual_patterns(self) -> None:
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

        def add_patterns(input_dict: dict) -> None:
            """
            It creates a  list of dictionaries that can be used for a spaCy entity ruler

            :param input_dict: a dictionary
            :type input_dict: dict
            """
            for key in input_dict:
                if self.match_key == "LEMMA":
                    words = [
                        "".join(
                            [
                                token.lemma_ + token.whitespace_
                                if token.lemma_
                                else token.text
                                for token in doc
                            ]
                        )
                        for doc in self.nlp.pipe(input_dict[key])
                    ]
                else:
                    words = input_dict[key]
                for word in words:
                    if word != key:
                        specific_match_rule = {**self.match_rule}
                        word_parts = self._split_word(word)
                        if len(word_parts) > 1:
                            operators = [" ", "-"]
                        else:
                            operators = [""]

                        for op in operators:
                            if self.case_sensitive:
                                specific_match_rule[self.match_key] = f"{op}".join(
                                    word_parts
                                )
                            else:
                                specific_match_rule[self.match_key] = {
                                    "regex": r"(?i)"
                                    + re.escape(f"{op}".join(word_parts))
                                }

                            patterns.append(
                                {
                                    "label": key.upper(),
                                    "pattern": [
                                        specific_match_rule,
                                    ],
                                    "id": f"{word}_{op}_individual",
                                }
                            )

                            if self.include_compound_words:
                                compound_rule = {
                                    "DEP": {"IN": ["amod", "compound"]},
                                    "OP": "*",
                                }
                                patterns.append(
                                    {
                                        "label": key.upper(),
                                        "pattern": [
                                            compound_rule,
                                            specific_match_rule,
                                            compound_rule,
                                        ],
                                        "id": f"{word}_{op}_compound",
                                    }
                                )

        add_patterns(self.data)
        if self.json_path:
            with open(self.json_path, "w") as f:
                json.dump(patterns, f)
        self.ruler = self.nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
        self.ruler.add_patterns(patterns)

    def __call__(self, doc: Doc) -> Doc:
        """
        It takes a doc object and assigns a score to each entity in the doc object

        :param doc: Doc
        :type doc: Doc
        """
        if self.ent_score:
            doc = self.assign_score_to_entities(doc)
        return doc

    def pipe(self, stream, batch_size=128) -> Doc:
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

    def assign_score_to_entities(self, doc: Doc) -> Doc:
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
                ent_text = ent.text

                # get word part representations
                if self._check_presence_vocab(ent_text):
                    entity = [self._check_presence_vocab(ent_text)]
                else:
                    entity = []
                    for part in self._split_word(ent_text):
                        present_part = self._check_presence_vocab(part)
                        if present_part:
                            entity.append(present_part)

                # get concepts to match
                concept = self.concept_data.get(ent.label_, None)

                # compare set similarities
                if entity and concept:
                    ent._.ent_score = self.kv.n_similarity(entity, concept)
                else:
                    ent._.ent_score = 0
                    if self.verbose:
                        if f"{ent_text}_{concept}" not in self.log_cache["key_word"]:
                            logger.warning(
                                f"Entity ´{ent.text}´ and/or label ´{concept}´ not"
                                " found in vector model. Nothing to compare to, so"
                                " setting ent._.ent_score to 0."
                            )
                            self.log_cache["key_word"].append(f"{ent_text}_{concept}")
            else:
                ent._.ent_score = 0
                if self.verbose:
                    if ent.text not in self.log_cache["word"]:
                        logger.warning(
                            f"Entity ´{ent.text}´ not found in vector model. Nothing to"
                            " compare to, so setting ent._.ent_score to 0."
                        )
                        self.log_cache["word"].append(ent.text)
        doc.ents = ents
        return doc

    def set_concept_dict(self):
        self.concept_data = {k.upper(): v for k, v in self.data.items()}
        for ent_label in self.concept_data:
            concept = []
            for word in self.concept_data[ent_label]:
                present_word = self._check_presence_vocab(word)
                if present_word:
                    concept.append(present_word)
            self.concept_data[ent_label] = concept

    def _split_word(self, word: str) -> List[str]:
        """
        It splits a word into a list of subwords, using the word delimiter

        :param word: str
        :type word: str
        :return: A list of strings or any.
        """
        return re.split(f"[{re.escape(self.word_delimiter)}]+", word)

    def _check_presence_vocab(self, word: str) -> str:
        """
        If the word is not lowercase and the case_sensitive flag is set to False, then check if the lowercase version of
        the word is in the vocabulary. If it is, return the lowercase version of the word. Otherwise, return the word
        itself

        :param word: The word to check for presence in the vocabulary
        :type word: str
        :return: The word itself if it is present in the vocabulary, otherwise the word with the highest probability of
        being the word that was intended.
        """
        if not word.islower() and not self.case_sensitive:
            present_word = self.__check_presence_vocab(word.lower())
            if present_word:
                return present_word
        return self.__check_presence_vocab(word)

    def __check_presence_vocab(self, word: str) -> str:
        """
        If the word is in the vocabulary, return the word. If not, replace spaces and dashes with the word delimiter and
        check if the new word is in the vocabulary. If so, return the new word

        :param word: str - the word to check
        :type word: str
        :return: The word or the check_word
        """
        if word in self.kv:
            return word

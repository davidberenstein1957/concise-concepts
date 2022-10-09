# -*- coding: utf-8 -*-
from typing import List, Union

from spacy.language import Language

from .conceptualizer import Conceptualizer


@Language.factory(
    "concise_concepts",
    default_config={
        "data": None,
        "topn": [],
        "model_path": None,
        "word_delimiter": "_",
        "ent_score": False,
        "exclude_pos": [
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
        ],
        "exclude_dep": [],
        "include_compound_words": False,
        "case_sensitive": False,
        "json_path": "./matching_patterns.json",
        "verbose": True,
    },
)
def make_concise_concepts(
    nlp: Language,
    name: str,
    data: Union[dict, list],
    topn: list,
    model_path: Union[str, None],
    word_delimiter: str,
    ent_score: bool,
    exclude_pos: List[str],
    exclude_dep: List[str],
    include_compound_words: bool,
    case_sensitive: bool,
    json_path: str,
    verbose: bool,
):
    return Conceptualizer(
        nlp=nlp,
        name=name,
        data=data,
        topn=topn,
        model_path=model_path,
        word_delimiter=word_delimiter,
        ent_score=ent_score,
        exclude_pos=exclude_pos,
        exclude_dep=exclude_dep,
        include_compound_words=include_compound_words,
        case_sensitive=case_sensitive,
        json_path=json_path,
        verbose=verbose,
    )

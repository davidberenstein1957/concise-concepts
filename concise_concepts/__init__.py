# -*- coding: utf-8 -*-
from typing import Union

from spacy.language import Language

from .conceptualizer import ConceptualSpacy


@Language.factory(
    "concise_concepts",
    default_config={"data": None, "topn": [], "model_path": None, "ent_score": False},
)
def make_concise_concepts(
    nlp: Language, name: str, data: Union[dict, list], topn: list, model_path: Union[str, None], ent_score: bool
):
    return ConceptualSpacy(nlp=nlp, name=name, data=data, topn=topn, model_path=model_path, ent_score=ent_score)

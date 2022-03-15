import os
from typing import Union

from spacy.language import Language

from .conceptualizer import ConceptualSpacy


@Language.factory(
    "concise_concepts",
    default_config={
        "data": None,
        "topn": []
    },
)
def make_concise_concepts(
    nlp: Language,
    name: str,
    data: Union[dict, list],
    topn: list
):  
    return ConceptualSpacy(
        nlp=nlp,
        name=name,
        data=data,
        topn=topn
    )

# -*- coding: utf-8 -*-
import spacy
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

import concise_concepts  # noqa: F401

data = {"human": ["trees"], "interface": ["computer"]}

text = (
    "believe me, it's the slowest mobile I saw. Don't go on screen and Battery, it is"
    " an extremely slow mobile phone and takes ages to open and navigate. Forget about"
    " heavy use, it can't handle normal regular use. I made a huge mistake but pls"
    " don't buy this mobile. It's only a few months and I am thinking to change it. Its"
    " dam SLOW SLOW SLOW."
)

model = Word2Vec(
    sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4
)
model.save("word2vec.model")
model_path = "word2vec.model"

nlp = spacy.blank("en")
nlp.add_pipe("concise_concepts", config={"data": data, "model_path": model_path})

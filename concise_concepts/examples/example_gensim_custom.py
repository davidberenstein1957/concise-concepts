# -*- coding: utf-8 -*-
import gensim.downloader as api
import spacy

import concise_concepts  # noqa: F401

from .data import data, text

model_path = "word2vec.model"
model = api.load("glove-twitter-25")
model.save(model_path)
nlp = spacy.blank("en")

nlp.add_pipe("concise_concepts", config={"data": data, "model_path": model_path})

doc = nlp(text)
print([(ent.text, ent.label_) for ent in doc.ents])

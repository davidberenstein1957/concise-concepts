# -*- coding: utf-8 -*-
import spacy

import concise_concepts  # noqa: F401

from .data import data, text

nlp = spacy.load("en_core_web_md")

nlp.add_pipe("concise_concepts", config={"data": data})

doc = nlp(text)
print([(ent.text, ent.label_) for ent in doc.ents])

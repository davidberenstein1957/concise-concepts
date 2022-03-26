import spacy

import concise_concepts

from .data import data, text

nlp = spacy.load('en_core_web_lg')

nlp.add_pipe("concise_concepts", config={"data": data})

doc = nlp(text)
print([(ent.text, ent.label_) for ent in doc.ents])

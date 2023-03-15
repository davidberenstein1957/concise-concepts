# Concise Concepts
When wanting to apply NER to concise concepts, it is really easy to come up with examples, but pretty difficult to train an entire pipeline. Concise Concepts uses few-shot NER based on word embedding similarity to get you going
with easy! Now with entity scoring!


[![Python package](https://github.com/Pandora-Intelligence/concise-concepts/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/Pandora-Intelligence/concise-concepts/actions/workflows/python-package.yml)
[![Current Release Version](https://img.shields.io/github/release/pandora-intelligence/concise-concepts.svg?style=flat-square&logo=github)](https://github.com/pandora-intelligence/concise-concepts/releases)
[![pypi Version](https://img.shields.io/pypi/v/concise-concepts.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/concise-concepts/)
[![PyPi downloads](https://static.pepy.tech/personalized-badge/concise-concepts?period=total&units=international_system&left_color=grey&right_color=orange&left_text=pip%20downloads)](https://pypi.org/project/concise-concepts/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)


## Usage
This library defines matching patterns based on the most similar words found in each group, which are used to fill a [spaCy EntityRuler](https://spacy.io/api/entityruler). To better understand the rule definition, I recommend playing around with the [spaCy Rule-based Matcher Explorer](https://demos.explosion.ai/matcher).

### Tutorials
- [TechVizTheDataScienceGuy](https://www.youtube.com/c/TechVizTheDataScienceGuy) created a [nice tutorial](https://prakhar-mishra.medium.com/few-shot-named-entity-recognition-in-natural-language-processing-92d31f0d1143) on how to use it.

- [I](https://www.linkedin.com/in/david-berenstein-1bab11105/) created a [tutorial](https://www.rubrix.ml/blog/concise-concepts-rubrix/) in collaboration with Rubrix.

The section [Matching Pattern Rules](#matching-pattern-rules) expands on the construction, analysis and customization of these matching patterns.


# Install

```
pip install concise-concepts
```

# Quickstart

Take a look at the [configuration section](#configuration) for more info.

## Spacy Pipeline Component

Note that, [custom embedding models](#custom-embedding-models) are passed via `model_path`.

```python
import spacy
from spacy import displacy

import concise_concepts

data = {
    "fruit": ["apple", "pear", "orange"],
    "vegetable": ["broccoli", "spinach", "tomato"],
    "meat": ['beef', 'pork', 'turkey', 'duck']
}

text = """
    Heat the oil in a large pan and add the Onion, celery and carrots.
    Then, cook over a medium–low heat for 10 minutes, or until softened.
    Add the courgette, garlic, red peppers and oregano and cook for 2–3 minutes.
    Later, add some oranges and chickens. """

nlp = spacy.load("en_core_web_md", disable=["ner"])

nlp.add_pipe(
    "concise_concepts",
    config={
        "data": data,
        "ent_score": True,  # Entity Scoring section
        "verbose": True,
        "exclude_pos": ["VERB", "AUX"],
        "exclude_dep": ["DOBJ", "PCOMP"],
        "include_compound_words": False,
        "json_path": "./fruitful_patterns.json",
        "topn": (100,500,300)
    },
)
doc = nlp(text)

options = {
    "colors": {"fruit": "darkorange", "vegetable": "limegreen", "meat": "salmon"},
    "ents": ["fruit", "vegetable", "meat"],
}

ents = doc.ents
for ent in ents:
    new_label = f"{ent.label_} ({ent._.ent_score:.0%})"
    options["colors"][new_label] = options["colors"].get(ent.label_.lower(), None)
    options["ents"].append(new_label)
    ent.label_ = new_label
doc.ents = ents

displacy.render(doc, style="ent", options=options)
```
![](https://raw.githubusercontent.com/Pandora-Intelligence/concise-concepts/master/img/example.png)

## Standalone

This might be useful when iterating over few_shot training data when not wanting to reload larger models continuously.
Note that, [custom embedding models](#custom-embedding-models) are passed via `model`.

```python
import gensim
import spacy

from concise_concepts import Conceptualizer

model = gensim.downloader.load("fasttext-wiki-news-subwords-300")
nlp = spacy.load("en_core_web_sm")
data = {
    "disease": ["cancer", "diabetes", "heart disease", "influenza", "pneumonia"],
    "symptom": ["headache", "fever", "cough", "nausea", "vomiting", "diarrhea"],
}
conceptualizer = Conceptualizer(nlp, data, model)
conceptualizer.nlp("I have a headache and a fever.").ents

data = {
    "disease": ["cancer", "diabetes"],
    "symptom": ["headache", "fever"],
}
conceptualizer = Conceptualizer(nlp, data, model)
conceptualizer.nlp("I have a headache and a fever.").ents
```

# Configuration
## Matching Pattern Rules
A general introduction about the usage of matching patterns in the [usage section](#usage).
### Customizing Matching Pattern Rules
Even though the baseline parameters provide a decent result, the construction of these matching rules can be customized via the config passed to the spaCy pipeline.

 - `exclude_pos`: A list of POS tags to be excluded from the rule-based match.
 - `exclude_dep`: A list of dependencies to be excluded from the rule-based match.
 - `include_compound_words`:  If True, it will include compound words in the entity. For example, if the entity is "New York", it will also include "New York City" as an entity.
 - `case_sensitive`: Whether to match the case of the words in the text.


### Analyze Matching Pattern Rules
To motivate actually looking at the data and support interpretability, the matching patterns that have been generated are stored as `./main_patterns.json`. This behavior can be changed by using the `json_path` variable via the config passed to the spaCy pipeline.


## Fuzzy matching using `spaczz`

 - `fuzzy`: A boolean value that determines whether to use fuzzy matching

```python
data = {
    "fruit": ["apple", "pear", "orange"],
    "vegetable": ["broccoli", "spinach", "tomato"],
    "meat": ["beef", "pork", "fish", "lamb"]
}

nlp.add_pipe("concise_concepts", config={"data": data, "fuzzy": True})
```

## Most Similar Word Expansion

- `topn`: Use a specific number of words to expand over.

```python
data = {
    "fruit": ["apple", "pear", "orange"],
    "vegetable": ["broccoli", "spinach", "tomato"],
    "meat": ["beef", "pork", "fish", "lamb"]
}

topn = [50, 50, 150]

assert len(topn) == len

nlp.add_pipe("concise_concepts", config={"data": data, "topn": topn})
```

## Entity Scoring

- `ent_score`: Use embedding based word similarity to score entities against their groups

```python
import spacy
import concise_concepts

data = {
    "ORG": ["Google", "Apple", "Amazon"],
    "GPE": ["Netherlands", "France", "China"],
}

text = """Sony was founded in Japan."""

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("concise_concepts", config={"data": data, "ent_score": True, "case_sensitive": True})
doc = nlp(text)

print([(ent.text, ent.label_, ent._.ent_score) for ent in doc.ents])
# output
#
# [('Sony', 'ORG', 0.5207586), ('Japan', 'GPE', 0.7371268)]
```

## Custom Embedding Models

- `model_path`: Use custom `sense2vec.Sense2Vec`, `gensim.Word2vec` `gensim.FastText`, or `gensim.KeyedVectors`, or a pretrained model from [gensim](https://radimrehurek.com/gensim/downloader.html) library or a custom model path. For using a `sense2vec.Sense2Vec` take a look [here](https://github.com/explosion/sense2vec#pretrained-vectors).
- `model`: within [standalone usage](#standalone), it is possible to pass these models directly.

```python
data = {
    "fruit": ["apple", "pear", "orange"],
    "vegetable": ["broccoli", "spinach", "tomato"],
    "meat": ["beef", "pork", "fish", "lamb"]
}

# model from https://radimrehurek.com/gensim/downloader.html or path to local file
model_path = "glove-wiki-gigaword-300"

nlp.add_pipe("concise_concepts", config={"data": data, "model_path": model_path})
````

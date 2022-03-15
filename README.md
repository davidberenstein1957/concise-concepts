# Concise Concepts
When wanting to apply NER to concise concepts, it is really easy to come up with examples, but it takes some time to train an entire pipeline. Concise Concepts uses word similarity based on few-shots to get you going with easy!

# Install
``` pip install classy-classification```

# Quickstart
```
import spacy
import concise_concepts

data = {
    "fruit": ["apple", "pear", "orange"],
    "vegetable": ["broccoli", "spinach", "tomato"],
    "meat": ["chicken", "beef", "pork", "fish", "lamb"]
}

text = """
    Heat the oil in a large pan and add the Onion, celery and carrots. 
    Cook over a medium–low heat for 10 minutes, or until softened. 
    Add the courgette, garlic, red peppers and oregano and cook for 2–3 minutes.
    Later, add some oranges and chickens. """

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe("concise_concepts", config={"data": data})
doc = nlp(text)

print([(ent.text, ent.label_) for ent in doc.ents])
# Output:
#
# [('Onion', 'VEGETABLE'), ('Celery', 'VEGETABLE'), ('carrots', 'VEGETABLE'), 
#  ('garlic', 'VEGETABLE'), ('red peppers', 'VEGETABLE'), ('oranges', 'FRUIT'), 
#  ('chickens', 'MEAT')]


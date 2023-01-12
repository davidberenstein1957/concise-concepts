# -*- coding: utf-8 -*-
def test_spacy_embeddings():
    from concise_concepts.examples import example_spacy  # noqa: F401


def test_gensim_default():
    from concise_concepts.examples import example_gensim_default  # noqa: F401


def test_gensim_custom_path():
    from concise_concepts.examples import example_gensim_custom_path  # noqa: F401


def test_gensim_custom_model():
    from concise_concepts.examples import example_gensim_custom_model  # noqa: F401


def test_standalone_spacy():
    import spacy

    from concise_concepts import Conceptualizer

    nlp = spacy.load("en_core_web_md")
    data = {
        "disease": ["cancer", "diabetes", "heart disease", "influenza", "pneumonia"],
        "symptom": ["headache", "fever", "cough", "nausea", "vomiting", "diarrhea"],
    }
    conceptualizer = Conceptualizer(nlp, data)
    assert (
        list(conceptualizer.pipe(["I have a headache and a fever."]))[0].to_json()
        == list(conceptualizer.nlp.pipe(["I have a headache and a fever."]))[
            0
        ].to_json()
    )
    assert (
        conceptualizer("I have a headache and a fever.").to_json()
        == conceptualizer.nlp("I have a headache and a fever.").to_json()
    )

    data = {
        "disease": ["cancer", "diabetes"],
        "symptom": ["headache", "fever"],
    }
    conceptualizer = Conceptualizer(nlp, data)


def test_standalone_gensim():
    import gensim
    import spacy

    from concise_concepts import Conceptualizer

    model_path = "glove-twitter-25"
    model = gensim.downloader.load(model_path)
    nlp = spacy.load("en_core_web_md")
    data = {
        "disease": ["cancer", "diabetes", "heart disease", "influenza", "pneumonia"],
        "symptom": ["headache", "fever", "cough", "nausea", "vomiting", "diarrhea"],
    }
    conceptualizer = Conceptualizer(nlp, data, model=model)
    print(list(conceptualizer.pipe(["I have a headache and a fever."]))[0].ents)
    print(list(conceptualizer.nlp.pipe(["I have a headache and a fever."]))[0].ents)
    print(conceptualizer("I have a headache and a fever.").ents)
    print(conceptualizer.nlp("I have a headache and a fever.").ents)


def test_spaczz():
    # -*- coding: utf-8 -*-
    import spacy

    import concise_concepts  # noqa: F401
    from concise_concepts.examples.data import data, text, text_fuzzy

    nlp = spacy.load("en_core_web_md")

    nlp.add_pipe("concise_concepts", config={"data": data, "fuzzy": True})

    assert len(nlp(text).ents) == len(nlp(text_fuzzy).ents)


def test_sense2vec():
    # -*- coding: utf-8 -*-
    import requests
    import spacy

    import concise_concepts  # noqa: F401
    from concise_concepts.examples.data import data, text

    model_path = "s2v_old"
    # download .tar.gz file an URL
    # and extract it to a folder
    url = "https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz"
    r = requests.get(url, allow_redirects=True)
    open("s2v_reddit_2015_md.tar.gz", "wb").write(r.content)
    # extract tar.gz file
    filename = "s2v_reddit_2015_md.tar.gz"
    import tarfile

    tar = tarfile.open(filename, "r:gz")
    tar.extractall()
    tar.close()

    nlp = spacy.load("en_core_web_md")

    nlp.add_pipe("concise_concepts", config={"data": data, "model_path": model_path})

    assert len(nlp(text).ents)

import spacy

import concise_concepts

data = {
    "performance": ["mileage", "speed", "fuel", "capacity", "transmisson"],
    "engine": ["turbo", "cylinder", "gear", "engine"],
    "brakes": ["disc", "suspension", "brakes"],
    "dimensions": ["length", "width", "height", "seating", "doors"],
    "comfort": [
        "steering",
        "heater",
        "air",
        "accessory",
        "headrest",
        "charger",
        "luxury",
    ],
    "entertainment": ["radio", "speaker", "phone", "touch"],
    "safety": ["alarm", "theft", "warning", "safety"],
}

text = """
XUV 700 is first in class luxury SUV brought to you by Mahindra.
The 13-inch instrument cluster and attached infotainment panel
are just the best in class and add a classy look to the SUV.
The XUV 700 is a beast of a car with that panoramic sunroof,
and leather upholstery on the seats and doesn't compromise on the luxury of the passengers.
The adaptive cruise control is a top-end feature clearly not offered in any
other cars of this segment at this price point. Have always been a fan of the XUV 500
but the successor is a beast with the luxury of its own kind the road experience is pretty smooth
and the engine performs really well.
"""
model_path = "glove-wiki-gigaword-300"
nlp = spacy.load("en_core_web_lg", disable=["ner"])
nlp.add_pipe(
    "concise_concepts",
    config={"data": data, "model_path": model_path, "ent_score": True},
)

import pytest

from summarizer.text_processors.coreference_handler import CoreferenceHandler


@pytest.fixture()
def coreference_handler():
    return CoreferenceHandler()


def test_coreference_handler(coreference_handler):
    orig = "My sister has a dog. She loves him."
    resolved = "My sister has a dog. My sister loves a dog."
    result = coreference_handler.process(orig, min_length=2)
    assert " ".join(result) == resolved


def test_longer_coreference_handler(coreference_handler):
    orig = "My sister has a dog. My sister loves him. John Smith called from London, he said it's raining in the city."
    resolved = "My sister has a dog. My sister loves a dog. " \
               "John Smith called from London, John Smith said it's raining in London."
    result = coreference_handler.process(orig, min_length=2)
    assert " ".join(result) == resolved


def test_possessive_coreference_handler(coreference_handler):
    orig = "The government announced the new policy on Wednesday. " \
           "Their spokesperson, Angela Smith, said that they had listened to all proposals."
    resolved = "The government announced the new policy on Wednesday. " \
               "The government\u2019s spokesperson, Angela Smith, said that The government had listened to all proposals."
    result = coreference_handler.process(orig, min_length=2)
    assert " ".join(result) == resolved


def test_possessive_coreference_handler_2(coreference_handler):
    orig = "Next, the cat sat on the mat. We tickled its nose."
    resolved = "Next, the cat sat on the mat. We tickled the cat\u2019s nose."
    result = coreference_handler.process(orig, min_length=2)
    assert " ".join(result) == resolved


def test_overlapping_coreference_handler(coreference_handler):
    orig = """Fidel Castro led a communist revolution that toppled the Cuban government in 1959, after which he declared himself prime minister. He held the title until 1976, when it was abolished and he became head of the Communist Party and president of the council of state and the council of ministers. With his health failing, Castro handed power to his brother, Raúl, in 2006. He died in 2016."""
    resolved = "Fidel Castro led a communist revolution that toppled the Cuban government in 1959, after which Fidel Castro declared Fidel Castro prime minister. Fidel Castro held himself prime minister until 1976, when himself prime minister was abolished and Fidel Castro became head of the Communist Party and president of the council of state and the council of ministers. With Fidel Castro\u2019s health failing, Fidel Castro handed power to Fidel Castro\u2019s brother, Raúl, in 2006. Fidel Castro died in 2016."
    result = coreference_handler.process(orig, min_length=2)
    assert " ".join(result) == resolved

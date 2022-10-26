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

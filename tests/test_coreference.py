import pytest

from summarizer.text_processors.coreference_handler import CoreferenceHandler


@pytest.fixture()
def coreference_handler():
    return CoreferenceHandler()


def test_coreference_handler(coreference_handler):
    orig = "My sister has a dog. She loves him."
    resolved = ['My sister', 'She', 'a dog', 'him']
    result = coreference_handler.process(orig, min_length=2)
    assert result == resolved

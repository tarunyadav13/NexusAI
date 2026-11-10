import pytest

from unittest.mock import MagicMock

from nexusai.embeddings.generator import EmbeddingGenerator


@pytest.fixture
def generator():

    obj = EmbeddingGenerator.__new__(
        EmbeddingGenerator
    )

    obj.dimension = 768

    obj.model = "gemini-embedding-2"

    obj.client = MagicMock()

    return obj


def test_embedding_dimension_validation(generator):

    fake_embedding = MagicMock()

    fake_embedding.values = [0.1] * 768

    fake_response = MagicMock()

    fake_response.embeddings = [fake_embedding]

    generator.client.models.embed_content.return_value = (
        fake_response
    )

    result = generator.embed_query(
        "What is artificial intelligence?"
    )

    assert len(result) == 768


def test_empty_query(generator):

    with pytest.raises(ValueError):

        generator.embed_query("")


def test_empty_documents(generator):

    result = generator.embed_documents([])

    assert result == []
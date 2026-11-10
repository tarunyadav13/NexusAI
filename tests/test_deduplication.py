from nexusai.ingestion.deduplication import (
    DocumentDeduplicator
)


def test_hash_generation(tmp_path):

    file = tmp_path / "sample.txt"

    file.write_text(
        "NexusAI testing"
    )

    deduplicator = DocumentDeduplicator(
        tmp_path / "manifest.json"
    )

    hash1 = deduplicator.calculate_hash(file)

    hash2 = deduplicator.calculate_hash(file)

    assert hash1 == hash2


def test_duplicate_detection(tmp_path):

    file = tmp_path / "sample.txt"

    file.write_text("NexusAI")

    deduplicator = DocumentDeduplicator(
        tmp_path / "manifest.json"
    )

    fingerprint = deduplicator.calculate_hash(file)

    assert deduplicator.find_duplicate(
        fingerprint
    ) is None

    deduplicator.register_document(

        fingerprint=fingerprint,

        document_id="doc123",

        filename="sample.txt",

        source_path=str(file),

        output_path="output.jsonl"

    )

    result = deduplicator.find_duplicate(
        fingerprint
    )

    assert result is not None

    assert result["document_id"] == "doc123"


def test_manifest_persistence(tmp_path):

    file = tmp_path / "sample.txt"

    file.write_text("NexusAI")

    manifest = tmp_path / "manifest.json"

    deduplicator = DocumentDeduplicator(
        manifest
    )

    fingerprint = deduplicator.calculate_hash(file)

    deduplicator.register_document(

        fingerprint,

        "doc123",

        "sample.txt",

        str(file),

        "output.jsonl"

    )

    # Create a new instance to simulate restart

    new_deduplicator = DocumentDeduplicator(
        manifest
    )

    assert new_deduplicator.find_duplicate(
        fingerprint
    ) is not None
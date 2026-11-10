import json

from nexusai.ingestion.pipeline import IngestionPipeline


def test_complete_ingestion(tmp_path):

    raw_dir = tmp_path / "raw"

    processed_dir = tmp_path / "processed"

    raw_dir.mkdir()

    processed_dir.mkdir()

    file = raw_dir / "report.txt"

    file.write_text(
        "Artificial intelligence and "
        "retrieval augmented generation. " * 100,
        encoding="utf-8"
    )

    pipeline = IngestionPipeline(

        chunk_size=200,

        chunk_overlap=40,

        output_dir=processed_dir

    )

    result = pipeline.ingest(file)

    assert result.status == "processed"

    assert result.chunk_count > 0

    assert result.document_id is not None

    assert result.output_path is not None


def test_duplicate_ingestion(tmp_path):

    raw_dir = tmp_path / "raw"

    processed_dir = tmp_path / "processed"

    raw_dir.mkdir()

    processed_dir.mkdir()

    file = raw_dir / "report.txt"

    file.write_text(
        "NexusAI duplicate testing."
    )

    pipeline = IngestionPipeline(
        output_dir=processed_dir
    )

    first = pipeline.ingest(file)

    second = pipeline.ingest(file)

    assert first.status == "processed"

    assert second.status == "duplicate"

    assert first.document_id == second.document_id


def test_output_jsonl(tmp_path):

    file = tmp_path / "report.txt"

    file.write_text(
        "NexusAI is a document intelligence platform."
    )

    pipeline = IngestionPipeline(
        output_dir=tmp_path / "processed"
    )

    result = pipeline.ingest(file)

    output_path = result.output_path

    assert output_path is not None

    with open(
        output_path,
        encoding="utf-8"
    ) as output:

        lines = output.readlines()

    assert len(lines) > 0

    chunk = json.loads(lines[0])

    assert "chunk_id" in chunk

    assert "document_id" in chunk

    assert "text" in chunk

    assert "metadata" in chunk


def test_invalid_document(tmp_path):

    file = tmp_path / "missing.txt"

    pipeline = IngestionPipeline(
        output_dir=tmp_path / "processed"
    )

    result = pipeline.ingest(file)

    assert result.status == "failed"
import hashlib
import json
import logging
import threading

from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class DocumentDeduplicator:
    """
    Detects duplicate documents using SHA-256 fingerprints.

    Maintains a persistent JSON manifest.
    """

    def __init__(
        self,
        manifest_path: str | Path = (
            "data/processed/ingestion_manifest.json"
        )
    ):

        self.manifest_path = Path(manifest_path)

        self.manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self._lock = threading.RLock()

        self.manifest = self._load_manifest()

    def calculate_hash(
        self,
        file_path: str | Path
    ) -> str:

        path = Path(file_path)

        sha256 = hashlib.sha256()

        with path.open("rb") as file:

            for block in iter(
                lambda: file.read(1024 * 1024),
                b""
            ):

                sha256.update(block)

        return sha256.hexdigest()

    def _load_manifest(self) -> dict[str, Any]:

        if not self.manifest_path.exists():
            return {}

        try:

            with self.manifest_path.open(
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

                if not isinstance(data, dict):
                    raise ValueError(
                        "Manifest must contain a JSON object."
                    )

                return data

        except Exception as exc:

            raise RuntimeError(
                f"Unable to load ingestion manifest: {exc}"
            ) from exc

    def find_duplicate(
        self,
        fingerprint: str
    ) -> dict[str, Any] | None:

        with self._lock:

            return self.manifest.get(fingerprint)

    def register_document(
        self,
        fingerprint: str,
        document_id: str,
        filename: str,
        source_path: str,
        output_path: str
    ) -> None:

        with self._lock:

            if fingerprint in self.manifest:
                raise ValueError(
                    "Document fingerprint already registered."
                )

            self.manifest[fingerprint] = {

                "document_id": document_id,

                "filename": filename,

                "source_path": source_path,

                "output_path": output_path

            }

            self._save_manifest()

    def _save_manifest(self) -> None:

        temporary_path = self.manifest_path.with_suffix(
            ".tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.manifest,
                file,
                indent=4,
                ensure_ascii=False
            )

        temporary_path.replace(
            self.manifest_path
        )
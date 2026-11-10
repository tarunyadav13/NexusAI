import hashlib
import ipaddress
import socket
import logging

from urllib.parse import urlparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from nexusai.ingestion.schemas import (
    LoadedDocument,
    ExtractedPage
)

from nexusai.ingestion.exceptions import (
    DocumentExtractionError,
    EmptyDocumentError
)


logger = logging.getLogger(__name__)


class URLLoader:

    MAX_CONTENT_SIZE = 5 * 1024 * 1024

    USER_AGENT = (
        "Mozilla/5.0 (compatible; NexusAI/1.0; "
        "+https://example.com)"
    )

    def _validate_url(self, url):

        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only HTTP and HTTPS URLs are allowed.")

        if not parsed.hostname:
            raise ValueError("Invalid URL.")

        if parsed.username or parsed.password:
            raise ValueError("URLs containing credentials are not allowed.")

        hostname = parsed.hostname.lower()

        if hostname in ("localhost", "localhost.localdomain"):
            raise ValueError("Local URLs are not allowed.")

        try:
            addresses = socket.getaddrinfo(
                hostname,
                None
            )

            for address in addresses:

                ip = ipaddress.ip_address(address[4][0])

                if not ip.is_global:
                    raise ValueError(
                        "Private or non-public IP addresses are not allowed."
                    )

        except socket.gaierror as exc:
            raise ValueError("Unable to resolve URL hostname.") from exc

    def load(self, url: str) -> LoadedDocument:

        url = url.strip()

        self._validate_url(url)

        try:

            response = requests.get(
                url,
                headers={
                    "User-Agent": self.USER_AGENT
                },
                timeout=(10, 30),
                stream=True,
                allow_redirects=False
            )

            if response.is_redirect or response.is_permanent_redirect:
                raise DocumentExtractionError(
                    "Redirected URLs are not supported. "
                    "Please enter the final webpage URL."
                )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type",
                ""
            ).lower()

            if "text/html" not in content_type:
                raise DocumentExtractionError(
                    "URL must point to an HTML webpage."
                )

            content = response.content

            if len(content) > self.MAX_CONTENT_SIZE:
                raise DocumentExtractionError(
                    "Webpage exceeds the 5 MB extraction limit."
                )

            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            for element in soup([
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside",
                "noscript",
                "form"
            ]):
                element.decompose()

            title = (
                soup.title.get_text(strip=True)
                if soup.title
                else url
            )

            main = (
                soup.find("main")
                or soup.find("article")
                or soup.body
                or soup
            )

            text = main.get_text(
                separator="\n",
                strip=True
            )

            if not text.strip():
                raise EmptyDocumentError(
                    "No readable webpage content found."
                )

            parsed = urlparse(url)

            filename = (
                parsed.netloc.replace(":", "_")
                + "_"
                + hashlib.sha256(
                    url.encode("utf-8")
                ).hexdigest()[:10]
                + ".html"
            )

            return LoadedDocument(
                filename=filename,
                source_path=url,
                file_type="url",
                pages=[
                    ExtractedPage(
                        page_number=1,
                        text=f"Title: {title}\n\n{text}",
                        metadata={
                            "source_type": "webpage",
                            "url": url,
                            "title": title
                        }
                    )
                ]
            )

        except (
            DocumentExtractionError,
            EmptyDocumentError
        ):
            raise

        except Exception as exc:

            logger.exception(
                "Webpage extraction failed: %s",
                url
            )

            raise DocumentExtractionError(
                f"Unable to extract webpage: {exc}"
            ) from exc
import re
import unicodedata


class TextCleaner:
    """
    Cleans and normalizes extracted document text.
    """

    def clean(self, text: str) -> str:

        if not text:
            return ""

        # Normalize Unicode characters
        text = unicodedata.normalize(
            "NFKC",
            text
        )

        # Remove null bytes
        text = text.replace("\x00", "")

        # Remove unwanted control characters
        text = "".join(
            char
            for char in text
            if char in "\n\t" or not unicodedata.category(
                char
            ).startswith("C")
        )

        # Fix hyphenated line breaks
        text = re.sub(
            r"(?<=\w)-\s*\n\s*(?=\w)",
            "",
            text
        )

        # Normalize horizontal whitespace
        text = re.sub(
            r"[^\S\n]+",
            " ",
            text
        )

        # Remove spaces around newlines
        text = re.sub(
            r" *\n *",
            "\n",
            text
        )

        # Limit excessive blank lines
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()
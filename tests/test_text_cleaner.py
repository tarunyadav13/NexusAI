from nexusai.ingestion.text_cleaner import TextCleaner


def test_remove_extra_spaces():

    cleaner = TextCleaner()

    result = cleaner.clean(
        "Hello     World"
    )

    assert result == "Hello World"


def test_remove_excessive_newlines():

    cleaner = TextCleaner()

    result = cleaner.clean(
        "Hello\n\n\n\nWorld"
    )

    assert result == "Hello\n\nWorld"


def test_remove_null_bytes():

    cleaner = TextCleaner()

    result = cleaner.clean(
        "Hello\x00World"
    )

    assert result == "HelloWorld"


def test_fix_hyphenated_line_break():

    cleaner = TextCleaner()

    result = cleaner.clean(
        "artifi-\ncial intelligence"
    )

    assert result == "artificial intelligence"


def test_empty_text():

    cleaner = TextCleaner()

    assert cleaner.clean("") == ""
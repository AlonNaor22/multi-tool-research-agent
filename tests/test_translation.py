"""Tests for src/tools/translation_tool.py — text translation."""

import pytest
from unittest.mock import patch, MagicMock


class TestTranslation:
    """Test translation tool with mocked deep-translator."""

    def test_pipe_format_three_parts(self):
        with patch("src.tools.translation_tool._do_translate", return_value="Hola mundo"):
            from src.tools.translation_tool import translate_text
            result = translate_text("Hello world | en | es")

            assert "Hola mundo" in result
            assert "en" in result
            assert "es" in result

    def test_pipe_format_two_parts(self):
        with patch("src.tools.translation_tool._do_translate", return_value="Bonjour le monde"):
            from src.tools.translation_tool import translate_text
            result = translate_text("Hello world | to french")

            assert "Bonjour le monde" in result

    def test_natural_format(self):
        with patch("src.tools.translation_tool._do_translate", return_value="Hallo Welt"):
            from src.tools.translation_tool import translate_text
            result = translate_text("Hello world to german")

            assert "Hallo Welt" in result

    def test_language_aliases(self):
        from src.tools.translation_tool import _normalize_language
        assert _normalize_language("english") == "en"
        assert _normalize_language("spanish") == "es"
        assert _normalize_language("japanese") == "ja"
        assert _normalize_language("auto") == "auto"

    def test_empty_input(self):
        from src.tools.translation_tool import translate_text
        result = translate_text("")
        assert "Error" in result

    def test_help_command(self):
        from src.tools.translation_tool import translate_text
        result = translate_text("help")
        assert "FORMAT" in result

    def test_handles_translation_error(self):
        with patch("src.tools.translation_tool._do_translate",
                   side_effect=Exception("Translation API failed")):
            from src.tools.translation_tool import translate_text
            result = translate_text("Hello | en | es")

            assert "Error" in result

    def test_unparseable_input(self):
        from src.tools.translation_tool import translate_text
        result = translate_text("just some random text without target")
        assert "Error" in result or "Could not parse" in result

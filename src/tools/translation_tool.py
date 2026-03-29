"""Translation tool for the research agent.

Translates text between languages using Google Translate via deep-translator.
Free, no API key required, supports 100+ languages.
"""

import re
import asyncio
from langchain_core.tools import Tool

from src.utils import async_retry_on_error, make_sync


# Common language name -> code mapping for user-friendly input
LANGUAGE_ALIASES = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
    "chinese": "zh-CN", "japanese": "ja", "korean": "ko", "arabic": "ar",
    "hindi": "hi", "turkish": "tr", "polish": "pl", "swedish": "sv",
    "danish": "da", "norwegian": "no", "finnish": "fi", "greek": "el",
    "hebrew": "he", "thai": "th", "vietnamese": "vi", "indonesian": "id",
    "czech": "cs", "romanian": "ro", "hungarian": "hu", "ukrainian": "uk",
    "persian": "fa", "farsi": "fa", "bengali": "bn", "tamil": "ta",
    "telugu": "te", "malay": "ms", "filipino": "tl", "swahili": "sw",
    "catalan": "ca", "croatian": "hr", "serbian": "sr", "slovak": "sk",
    "slovenian": "sl", "bulgarian": "bg", "latvian": "lv", "lithuanian": "lt",
    "estonian": "et", "maltese": "mt", "icelandic": "is", "welsh": "cy",
    "irish": "ga", "afrikaans": "af", "albanian": "sq", "amharic": "am",
    "armenian": "hy", "azerbaijani": "az", "basque": "eu", "belarusian": "be",
    "bosnian": "bs", "burmese": "my", "georgian": "ka", "kazakh": "kk",
    "khmer": "km", "lao": "lo", "macedonian": "mk", "mongolian": "mn",
    "nepali": "ne", "urdu": "ur", "uzbek": "uz",
}


def _normalize_language(lang: str) -> str:
    """Convert a language name or code to a code accepted by GoogleTranslator."""
    lang = lang.strip().lower()
    if lang == "auto":
        return "auto"
    return LANGUAGE_ALIASES.get(lang, lang)


@async_retry_on_error(max_retries=2, delay=1.0)
async def _async_do_translate(text: str, source: str, target: str) -> str:
    """Perform the translation using deep-translator, offloaded to a thread."""
    from deep_translator import GoogleTranslator

    def _translate():
        translator = GoogleTranslator(source=source, target=target)
        return translator.translate(text)

    return await asyncio.to_thread(_translate)


async def translate_text(input_str: str) -> str:
    """
    Translate text between languages.

    Supports formats:
    - "Hello world | en | es"           (text | source | target)
    - "Hello world | to spanish"        (auto-detect source)
    - "Hello world to french"           (auto-detect source, natural format)

    Args:
        input_str: Text with language specification

    Returns:
        Translated text or error message
    """
    input_str = input_str.strip()

    if not input_str:
        return "Error: Empty translation request"

    if input_str.lower() in ("help", "?"):
        return _get_help()

    # Try pipe-delimited format first: "text | source | target"
    pipe_parts = [p.strip() for p in input_str.split("|")]

    if len(pipe_parts) == 3:
        text, source, target = pipe_parts
        source = _normalize_language(source)
        target = _normalize_language(target)
    elif len(pipe_parts) == 2:
        text = pipe_parts[0]
        target_part = pipe_parts[1]
        # Handle "to <language>" or just "<language>"
        if target_part.lower().startswith("to "):
            target_part = target_part[3:].strip()
        source = "auto"
        target = _normalize_language(target_part)
    else:
        # Try natural format: "text to <language>"
        to_match = re.match(r'(.+?)\s+to\s+(\w+)\s*$', input_str, re.IGNORECASE)
        if to_match:
            text = to_match.group(1).strip()
            source = "auto"
            target = _normalize_language(to_match.group(2))
        else:
            return (
                "Error: Could not parse translation request.\n"
                "Use format: 'text | source_lang | target_lang' or 'text to target_lang'\n"
                "Example: 'Hello world | en | es' or 'Hello world to spanish'"
            )

    if not text:
        return "Error: No text provided for translation"

    try:
        result = await _async_do_translate(text, source, target)
        source_label = source if source != "auto" else "auto-detected"
        return (
            f"Translation ({source_label} -> {target}):\n\n"
            f"{result}"
        )
    except Exception as e:
        return f"Error translating text: {str(e)}"


def _get_help() -> str:
    """Return help text."""
    common = ", ".join(f"{name} ({code})" for name, code in
                       list(LANGUAGE_ALIASES.items())[:20])
    return f"""Translation Tool Help:

FORMAT:
  Hello world | en | es              (text | source | target)
  Hello world | to spanish           (auto-detect source)
  Hello world to french              (natural format, auto-detect)

LANGUAGES (100+ supported, showing common ones):
  {common}, ...

TIPS:
  - Use language names (english, spanish) or codes (en, es)
  - Source language "auto" auto-detects the input language
  - Works with paragraphs of text, not just single sentences

EXAMPLES:
  "Bonjour le monde | fr | en"
  "Hello world to japanese"
  "The weather is nice today | to german" """


# Create the LangChain Tool wrapper
translation_tool = Tool(
    name="translate",
    func=make_sync(translate_text),
    coroutine=translate_text,
    description=(
        "Translate text between 100+ languages using Google Translate. "
        "\n\nFORMAT: 'text | source | target', 'text | to language', 'text to language'"
        "\n\nEXAMPLES: 'Hello | en | es', 'Bonjour to english', 'text | to japanese'"
        "\n\nRETURNS: Translated text with source/target language info."
        "\n\nUSE FOR: Translating non-English sources, research in other languages."
    )
)

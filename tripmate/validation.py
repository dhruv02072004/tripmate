from __future__ import annotations
import re


class InputValidationError(ValueError):
    """Message is safe to show to the end user."""


def validate_user_query(query: object, max_chars: int) -> str:
    if not isinstance(query, str):
        raise InputValidationError("Please send your question as text.")
    cleaned = "".join(ch for ch in query if ch.isprintable() or ch in "\n\t")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise InputValidationError(
            "Your message is empty. Ask me about visas, weather, packing, customs or safety "
            "for a destination."
        )
    if not any(ch.isalnum() for ch in cleaned):
        raise InputValidationError("I couldn't find any readable text in your message.")
    if len(cleaned) > max_chars:
        raise InputValidationError(
            f"Your message is too long ({len(cleaned)} characters; the limit is {max_chars}). "
            "Please shorten it."
        )
    return cleaned
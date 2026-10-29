"""PII Detection and Anonymization Engine."""
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class PIIMatch:
    entity_type: str
    original_text: str
    redacted_text: str
    start: int
    end: int


class PIIDetector:
    """Detects and redacts Personally Identifiable Information (PII) using robust regex patterns."""

    PATTERNS: Dict[str, str] = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "PHONE": r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "API_KEY": r"\b(?:sk-[a-zA-Z0-9]{20,48}|ghp_[a-zA-Z0-9]{36}|AIza[0-9A-Za-z-_]{35})\b",
        "IPV4": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
    }

    def __init__(self):
        self._compiled = {k: re.compile(v, re.IGNORECASE) for k, v in self.PATTERNS.items()}

    def detect(self, text: str) -> List[PIIMatch]:
        """Detect all PII entities in the provided text."""
        matches: List[PIIMatch] = []
        for entity_type, pattern in self._compiled.items():
            for m in pattern.finditer(text):
                original = m.group(0)
                matches.append(
                    PIIMatch(
                        entity_type=entity_type,
                        original_text=original,
                        redacted_text=f"<{entity_type}_REDACTED>",
                        start=m.start(),
                        end=m.end(),
                    )
                )
        return sorted(matches, key=lambda x: x.start)

    def redact(self, text: str) -> Tuple[str, List[PIIMatch]]:
        """Redacts sensitive PII entities and returns sanitized text along with audit matches."""
        matches = self.detect(text)
        if not matches:
            return text, []

        sanitized = text
        for m in sorted(matches, key=lambda x: x.start, reverse=True):
            sanitized = sanitized[: m.start] + m.redacted_text + sanitized[m.end :]
        return sanitized, matches

"""Input Guardrail validating queries against injection, PII, and unsafe content."""
import re
from dataclasses import dataclass, field
from typing import List, Optional
from .pii_detector import PIIDetector, PIIMatch


@dataclass
class InputGuardResult:
    passed: bool
    sanitized_text: str
    reasons: List[str] = field(default_factory=list)
    pii_matches: List[PIIMatch] = field(default_factory=list)
    risk_level: str = "LOW"


class InputGuard:
    """Validates user input queries before execution in the RAG pipeline."""

    INJECTION_PATTERNS = [
        r"(?i)\bignore\s+(all\s+)?(previous|prior)\s+instructions\b",
        r"(?i)\byou\s+are\s+now\s+(a\s+)?(dan|jailbroken|unrestricted|bypass)\b",
        r"(?i)\bsystem\s+prompt\s+override\b",
        r"(?i)\bdrop\s+table\b|\bexec\s*\(|\bselect\s+\*\s+from\b",
        r"(?i)\bprint\s+your\s+(initial|system)\s+instructions\b",
        r"(?i)\brevoke\s+all\s+guidelines\b",
    ]

    MAX_INPUT_LENGTH = 2000

    def __init__(self, pii_detector: Optional[PIIDetector] = None):
        self.pii_detector = pii_detector or PIIDetector()
        self._injection_compiled = [re.compile(p) for p in self.INJECTION_PATTERNS]

    def validate(self, query: str) -> InputGuardResult:
        reasons: List[str] = []
        risk_level = "LOW"

        if len(query.strip()) == 0:
            return InputGuardResult(
                passed=False,
                sanitized_text=query,
                reasons=["Empty query rejected"],
                risk_level="HIGH"
            )

        if len(query) > self.MAX_INPUT_LENGTH:
            reasons.append(f"Query exceeds max allowed length ({len(query)} > {self.MAX_INPUT_LENGTH})")
            risk_level = "HIGH"

        for pattern in self._injection_compiled:
            if pattern.search(query):
                reasons.append("Prompt injection / jailbreak signature detected")
                risk_level = "HIGH"
                break

        sanitized_query, pii_matches = self.pii_detector.redact(query)
        if pii_matches:
            reasons.append(f"PII detected and sanitized ({len(pii_matches)} entities)")
            if risk_level != "HIGH":
                risk_level = "MEDIUM"

        passed = (risk_level != "HIGH")
        return InputGuardResult(
            passed=passed,
            sanitized_text=sanitized_query if passed else query,
            reasons=reasons,
            pii_matches=pii_matches,
            risk_level=risk_level,
        )

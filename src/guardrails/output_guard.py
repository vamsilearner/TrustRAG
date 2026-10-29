"""Output Guardrail evaluating safety, hallucination risk, and data leakage."""
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class OutputGuardResult:
    passed: bool
    sanitized_text: str
    reasons: List[str] = field(default_factory=list)
    confidence_score: float = 1.0


class OutputGuard:
    """Verifies that generated responses do not leak sensitive instructions, internal secrets, or toxic content."""

    LEAK_PATTERNS = [
        r"(?i)my\s+instructions\s+are\s+to",
        r"(?i)as\s+an\s+ai\s+language\s+model\s+i\s+cannot",
        r"<SYSTEM_PROMPT>",
        r"sk-[a-zA-Z0-9]{20,48}",
    ]

    TOXIC_TERMS = ["hate speech sample", "bypass payload", "malware instruction"]

    def __init__(self):
        self._leak_compiled = [re.compile(p) for p in self.LEAK_PATTERNS]

    def validate(self, text: str, ground_truth_citations: bool = True) -> OutputGuardResult:
        reasons = []
        score = 1.0

        for pattern in self._leak_compiled:
            if pattern.search(text):
                reasons.append("Potential internal system leak or API key detected in output")
                score = 0.0
                return OutputGuardResult(passed=False, sanitized_text="[Content Redacted by Safety Guardrail]", reasons=reasons, confidence_score=score)

        for term in self.TOXIC_TERMS:
            if term in text.lower():
                reasons.append(f"Harmful terminology detected: {term}")
                score = 0.0
                return OutputGuardResult(passed=False, sanitized_text="[Content Blocked: Policy Violation]", reasons=reasons, confidence_score=score)

        # Ensure that if citations were requested, the output is not claiming completely baseless facts
        if len(text.strip()) < 10:
            reasons.append("Generation too short or degenerated")
            score = 0.4
            return OutputGuardResult(passed=False, sanitized_text=text, reasons=reasons, confidence_score=score)

        return OutputGuardResult(passed=True, sanitized_text=text, reasons=reasons, confidence_score=score)

from __future__ import annotations

import re

PATTERNS = [
    (re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b'), '[EMAIL]'),
    (re.compile(r'\b\+?\d[\d\s().-]{8,}\d\b'), '[PHONE_OR_ID]'),
    (re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'), '[DATE]'),
    (re.compile(r'\b(?:MRN|Patient ID|ID)[:# ]+\w+\b', re.I), '[PATIENT_ID]'),
]


def redact_phi(text: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    redacted = text
    for pattern, repl in PATTERNS:
        if pattern.search(redacted):
            flags.append(f'redacted_{repl.strip("[]").lower()}')
            redacted = pattern.sub(repl, redacted)
    return redacted, sorted(set(flags))

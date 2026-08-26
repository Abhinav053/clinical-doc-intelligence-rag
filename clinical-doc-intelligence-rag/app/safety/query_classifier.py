from __future__ import annotations

EMERGENCY_TERMS = {'chest pain', 'stroke', 'suicidal', 'shortness of breath', 'severe bleeding'}
DIAGNOSIS_TERMS = {'diagnose me', 'what disease do i have', 'do i have'}
DOSING_TERMS = {'dose', 'dosage', 'how much should i take'}


def classify_query(query: str) -> list[str]:
    q = query.lower()
    flags = []
    if any(term in q for term in EMERGENCY_TERMS):
        flags.append('emergency_or_urgent_symptoms')
    if any(term in q for term in DIAGNOSIS_TERMS):
        flags.append('diagnosis_request')
    if any(term in q for term in DOSING_TERMS):
        flags.append('medication_dosing_request')
    return flags

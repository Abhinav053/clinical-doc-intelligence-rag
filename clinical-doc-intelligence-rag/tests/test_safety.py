from app.safety.phi_redactor import redact_phi
from app.safety.query_classifier import classify_query


def test_redacts_email_and_date():
    text, flags = redact_phi('Patient email test@example.com DOB 01/02/1990')
    assert '[EMAIL]' in text
    assert '[DATE]' in text
    assert 'redacted_email' in flags


def test_classifies_diagnosis_request():
    assert 'diagnosis_request' in classify_query('Do I have pneumonia?')

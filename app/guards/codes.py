from typing import Literal

GuardFlag = Literal[
    "encoded_payload",
    "sensitive_input",
    "suspected_injection",
    "injection_deflected",
    "abuse_suspected",
    "hallucinated_number",
    "unlisted_contact",
    "foreign_identifier",
    "prohibited_promise",
    "threat",
    "artificial_urgency",
    "human_impersonation",
    "personal_advice",
    "third_party_disclosure",
    "tone_violation",
    "uncited_claim",
    "quote_not_in_source",
    "unsupported_sentence",
    "prompt_leak",
    "output_validation_failed",
]

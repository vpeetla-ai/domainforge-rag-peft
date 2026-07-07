from __future__ import annotations

import re
from typing import Any

# Keyword hints for baseline intent routing (S0/S1 before PEFT adapter)
INTENT_KEYWORDS: dict[str, list[str]] = {
    "track_order": [r"\btrack\b", r"\bwhere is my order\b", r"\bshipping status\b"],
    "get_refund": [r"\brefund\b", r"\bmoney back\b"],
    "check_refund_policy": [r"\brefund policy\b"],
    "recover_password": [r"\bpassword\b", r"\bforgot\b", r"\breset\b"],
    "payment_issue": [r"\bpayment\b", r"\bcharge\b", r"\bdeclined\b"],
    "contact_human_agent": [r"\bhuman\b", r"\bagent\b", r"\bsupervisor\b"],
    "contact_customer_service": [r"\bsupport\b", r"\bhelp\b", r"\bcustomer service\b"],
    "cancel_order": [r"\bcancel\b"],
    "delivery_period": [r"\blate\b", r"\bdelay\b"],
    "check_payment_methods": [r"\bpayment method\b", r"\bcard\b"],
}


def detect_intent(message: str, fallback: str = "contact_customer_service") -> str:
    text = message.lower()
    best_intent = fallback
    best_score = 0
    for intent, patterns in INTENT_KEYWORDS.items():
        score = sum(1 for pat in patterns if re.search(pat, text))
        if score > best_score:
            best_score = score
            best_intent = intent
    return best_intent


def intent_to_category(intent: str, sop_map: dict[str, Any]) -> str:
    for doc in sop_map.get("documents", []):
        if intent in doc.get("intent_tags", []):
            return doc.get("category", "CONTACT")
    return "CONTACT"

"""Step 3 - Query preprocessor.

Cleans text, normalizes whitespace/case-noise, expands common HR abbreviations,
and applies light spell correction. Deterministic and fast (no LLM) to keep
latency low; the heavier semantic rewrite happens in step 5.
"""
from __future__ import annotations

import re

# Common HR / company abbreviations -> expansions
ABBREVIATIONS = {
    "pf": "provident fund",
    "esi": "employee state insurance",
    "ctc": "cost to company",
    "wfh": "work from home",
    "ot": "overtime",
    "lop": "loss of pay",
    "cl": "casual leave",
    "sl": "sick leave",
    "el": "earned leave",
    "pl": "privilege leave",
    "ml": "maternity leave",
    "hr": "human resources",
    "appraisal": "performance appraisal",
    "emp": "employee",
    "doj": "date of joining",
    "dob": "date of birth",
    "ph": "public holiday",
    "lwp": "leave without pay",
}

# Tiny domain spell-correction map (extend as needed)
SPELL_FIXES = {
    "leav": "leave",
    "leaves": "leave",
    "policys": "policies",
    "policyy": "policy",
    "salery": "salary",
    "salray": "salary",
    "employee": "employee",
    "employe": "employee",
    "atendance": "attendance",
    "holyday": "holiday",
    "holy/day": "holiday",
    "resignaton": "resignation",
    "balence": "balance",
}


def _expand_abbreviations(text: str) -> str:
    def repl(match: re.Match) -> str:
        word = match.group(0)
        return ABBREVIATIONS.get(word.lower(), word)

    return re.sub(r"\b[a-zA-Z]{2,11}\b", repl, text)


def _spell_correct(text: str) -> str:
    def repl(match: re.Match) -> str:
        word = match.group(0)
        return SPELL_FIXES.get(word.lower(), word)

    return re.sub(r"\b[a-zA-Z]+\b", repl, text)


def preprocess(message: str) -> str:
    text = message.strip()
    # normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # strip control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # collapse repeated punctuation (e.g. "???" -> "?")
    text = re.sub(r"([?!.,])\1{1,}", r"\1", text)
    text = _spell_correct(text)
    text = _expand_abbreviations(text)
    return text.strip()

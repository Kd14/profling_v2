#!/usr/bin/env python3
"""
MyStera Health — ProfileIQ + Rules Scoring Engine (openEHR Edition)
====================================================================
Pipeline:
  1. Data Ingestion     — AQL queries per clinical domain
  2. Profile Synthesis  — Flat patient profile dict
  3. Rules Evaluation   — Expression-based rule engine (evaluates rule strings directly)
  4. Risk Classification — Confirmed / Unconfirmed / No Risk per disease
  5. Output             — Printed risk profiles + optional JSON export

Usage:
  python3 scripts/profile_and_score.py
  python3 scripts/profile_and_score.py --patient-id <ehr_id>
  python3 scripts/profile_and_score.py --output results.json
"""

import argparse, json, os, re
from datetime import date, datetime
from typing import Any
import requests
from requests.auth import HTTPBasicAuth

DEFAULT_URL      = os.getenv("EHRBASE_URL", "http://localhost:8080")
DEFAULT_USER     = os.getenv("EHRBASE_USER", "mystera")
DEFAULT_PASSWORD = os.getenv("EHRBASE_PASSWORD", "MysteraAdmin2024!")


# ══════════════════════════════════════════════════════════════════════════════
# RULES DEFINITIONS
# Expressed as evaluable strings — identical to the spec.
# &amp; and &gt; are decoded at evaluation time so these can be stored in XML/DB.
# ══════════════════════════════════════════════════════════════════════════════

rules_non_mod = {
    "rule_non_mod_angina_age":
        "((patient_gender == 'male' & patient_age > 45) | (patient_gender == 'female' & patient_age > 55))",
    "rule_non_mod_angina_ethnicity":
        "patient_ethnicity == 'South Asian'",
    "rule_non_mod_atherosclerosis_age":
        "patient_age > 65",
    "rule_non_mod_cardiogenicshock_age":
        "(patient_gender == 'female' & patient_age > 75)",
    "rule_non_mod_coronaryarterydisease_age":
        "(patient_gender == 'male' & patient_age > 65)",
}

rules_mod = {
    "rule_mod_angina":
        "patient_last_smoking == 'true' | patient_last_alcohol == 'true'",
    "rule_mod_atherosclerosis":
        "patient_last_smoking == 'true' | patient_last_alcohol == 'true'",
    "rule_mod_cardiogenicshock":
        "patient_last_smoking == 'true' | patient_last_alcohol == 'true'",
    "rule_mod_coronaryarterydisease":
        "patient_last_smoking == 'true' | patient_last_alcohol == 'true'",
}

rules_symptoms = {
    "rule_symptoms_angina":
        "patient_last_symptom_fatigue == 'true'",
    "rule_symptoms_atherosclerosis":
        "patient_last_symptom_chestpain == 'true'",
    "rule_symptoms_coronaryarterydisease":
        "patient_last_symptom_chestpain == 'true'",
}

rules_markers_blood = {
    "rule_blood_angina":
        "patient_last_blood_troponini > 0.04 | patient_last_blood_troponint > 0.01 | patient_last_vitals_creatinine > 5",
    "rule_blood_atherosclerosis":
        "patient_last_vitals_apob > 120 | patient_last_blood_ldlcholesterol > 160 | patient_last_blood_triglycerides > 200",
    "rule_blood_cardiogenicshock":
        "patient_last_vitals_creatinine > 1.5 | patient_last_blood_troponini > 0.04 | patient_last_blood_troponint > 0.01 | patient_last_vitals_bnp > 2000",
}

# Composite disease rules — source of truth for Confirmed Risk classification
rules_confirmed_disease = {
    "rule_confirmed_angina": (
        "(((("
        "(patient_gender == 'male' & patient_age > 45) | (patient_gender == 'female' & patient_age > 55))"
        " | (patient_ethnicity == 'South Asian')"
        " | (patient_last_smoking == 'true' | patient_last_alcohol == 'true'))"
        " & (patient_last_symptom_fatigue == 'true')"
        " & (patient_last_blood_troponini > 0.04 | patient_last_blood_troponint > 0.01 | patient_last_vitals_creatinine > 5)))"
    ),
    "rule_confirmed_atherosclerosis": (
        "((((patient_age > 65)"
        " | (patient_last_smoking == 'true' | patient_last_alcohol == 'true'))"
        " & (patient_last_symptom_chestpain == 'true')"
        " & (patient_last_vitals_apob > 120 | patient_last_blood_ldlcholesterol > 160 | patient_last_blood_triglycerides > 200)))"
    ),
    "rule_confirmed_cardiogenicshock": (
        "((((patient_gender == 'female' & patient_age > 75)"
        " | (patient_last_smoking == 'true' | patient_last_alcohol == 'true'))"
        " & (patient_last_vitals_creatinine > 1.5 | patient_last_blood_troponini > 0.04 | patient_last_blood_troponint > 0.01 | patient_last_vitals_bnp > 2000)))"
    ),
    "rule_confirmed_coronaryarterydisease": (
        "(((((patient_gender == 'male' & patient_age > 65)"
        " | (patient_last_smoking == 'true' | patient_last_alcohol == 'true'))"
        " & (patient_last_symptom_chestpain == 'true')"
        " & (patient_last_blood_troponini > 0.04 | patient_last_blood_troponint > 0.01 | patient_last_vitals_creatinine > 5))))"
    ),
}

# All rules in one registry for evaluation
ALL_RULES: dict[str, str] = {
    **rules_non_mod,
    **rules_mod,
    **rules_symptoms,
    **rules_markers_blood,
    **rules_confirmed_disease,
}


# ══════════════════════════════════════════════════════════════════════════════
# RULE EXPRESSION EVALUATOR
# Evaluates rule strings against a profile dict without using eval().
# Supports: ==, !=, >, <, >=, <=, &, |, (, ), string literals, numerics.
# ══════════════════════════════════════════════════════════════════════════════

def decode_rule(expr: str) -> str:
    """Normalise XML-encoded operators so rules can be stored in XML/DB."""
    return (expr
            .replace("&amp;", "&")
            .replace("&gt;",  ">")
            .replace("&lt;",  "<")
            .replace("&apos;", "'")
            .replace("&quot;", '"'))


def _tokenise(expr: str) -> list[str]:
    """Split rule expression into tokens."""
    token_re = re.compile(
        r"'[^']*'"          # string literal
        r"|>=|<=|==|!="     # two-char operators
        r"|[><!&|()]"       # single-char operators and parens
        r"|[\d.]+(?:\.\d+)?"  # numbers
        r"|[a-zA-Z_][a-zA-Z0-9_]*"  # identifiers
    )
    return token_re.findall(expr)


def _lookup(token: str, profile: dict) -> Any:
    """Resolve a token to a value: profile field, number, or string literal."""
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    if re.fullmatch(r'[\d]+(?:\.[\d]+)?', token):
        return float(token)
    return profile.get(token)  # None if missing


class _Parser:
    """
    Recursive descent parser for rule expressions.
    Grammar:
      expr   := or_expr
      or_expr  := and_expr ('|' and_expr)*
      and_expr := cmp_expr ('&' cmp_expr)*
      cmp_expr := '(' expr ')' | value op value
      op       := '==' | '!=' | '>' | '<' | '>=' | '<='
    """
    def __init__(self, tokens: list[str], profile: dict):
        self.tokens  = tokens
        self.pos     = 0
        self.profile = profile

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: str = None) -> str:
        tok = self.tokens[self.pos]
        if expected and tok != expected:
            raise ValueError(f"Expected '{expected}', got '{tok}' at pos {self.pos}")
        self.pos += 1
        return tok

    def parse(self) -> bool:
        return self.parse_or()

    def parse_or(self) -> bool:
        left = self.parse_and()
        while self.peek() == "|":
            self.consume("|")
            right = self.parse_and()
            left = left or right
        return left

    def parse_and(self) -> bool:
        left = self.parse_cmp()
        while self.peek() == "&":
            self.consume("&")
            right = self.parse_cmp()
            left = left and right
        return left

    def parse_cmp(self) -> bool:
        if self.peek() == "(":
            self.consume("(")
            result = self.parse_or()
            self.consume(")")
            return result

        lhs_tok = self.consume()
        op      = self.consume()
        rhs_tok = self.consume()

        lhs = _lookup(lhs_tok, self.profile)
        rhs = _lookup(rhs_tok, self.profile)

        # Normalise: if comparing numeric rhs to None lhs, treat lhs as 0
        if lhs is None:
            lhs = 0 if isinstance(rhs, float) else ""
        if rhs is None:
            rhs = 0 if isinstance(lhs, float) else ""

        try:
            if op == "==": return lhs == rhs
            if op == "!=": return lhs != rhs
            if op == ">":  return float(lhs) > float(rhs)
            if op == "<":  return float(lhs) < float(rhs)
            if op == ">=": return float(lhs) >= float(rhs)
            if op == "<=": return float(lhs) <= float(rhs)
        except (TypeError, ValueError):
            return False

        raise ValueError(f"Unknown operator: {op}")


def evaluate_rule(rule_expr: str, profile: dict) -> bool:
    """Evaluate a single rule expression string against a profile dict."""
    expr = decode_rule(rule_expr).strip()
    tokens = _tokenise(expr)
    if not tokens:
        return False
    try:
        parser = _Parser(tokens, profile)
        result = parser.parse()
        return bool(result)
    except Exception as e:
        print(f"    ⚠️  Rule eval error: {e} | expr: {expr[:80]}")
        return False


def evaluate_all_rules(profile: dict) -> dict[str, bool]:
    """Evaluate every rule in the registry against the profile."""
    return {name: evaluate_rule(expr, profile) for name, expr in ALL_RULES.items()}


# ══════════════════════════════════════════════════════════════════════════════
# RISK CLASSIFICATION
# Uses the confirmed disease rules as primary classifier.
# Unconfirmed = at least one non-mod OR mod rule for that disease fires.
# ══════════════════════════════════════════════════════════════════════════════

DISEASE_NAMES = ["angina", "atherosclerosis", "cardiogenicshock", "coronaryarterydisease"]

# Which subsidiary rules indicate unconfirmed risk per disease
UNCONFIRMED_RULES = {
    "angina": [
        "rule_non_mod_angina_age", "rule_non_mod_angina_ethnicity",
        "rule_mod_angina", "rule_symptoms_angina", "rule_blood_angina",
    ],
    "atherosclerosis": [
        "rule_non_mod_atherosclerosis_age",
        "rule_mod_atherosclerosis", "rule_symptoms_atherosclerosis", "rule_blood_atherosclerosis",
    ],
    "cardiogenicshock": [
        "rule_non_mod_cardiogenicshock_age",
        "rule_mod_cardiogenicshock", "rule_blood_cardiogenicshock",
    ],
    "coronaryarterydisease": [
        "rule_non_mod_coronaryarterydisease_age",
        "rule_mod_coronaryarterydisease", "rule_symptoms_coronaryarterydisease",
    ],
}

RISK_CONFIRMED   = "Confirmed Risk"
RISK_UNCONFIRMED = "Unconfirmed Risk"
RISK_NONE        = "No Risk"


def classify_risk(rule_results: dict[str, bool]) -> dict[str, str]:
    risk = {}
    for disease in DISEASE_NAMES:
        confirmed_rule = f"rule_confirmed_{disease}"

        # Confirmed: composite disease rule fired
        if rule_results.get(confirmed_rule, False):
            risk[disease] = RISK_CONFIRMED
            continue

        # Unconfirmed: at least one subsidiary rule fired
        subsidiary = UNCONFIRMED_RULES.get(disease, [])
        if any(rule_results.get(r, False) for r in subsidiary):
            risk[disease] = RISK_UNCONFIRMED
            continue

        risk[disease] = RISK_NONE

    return risk


# ══════════════════════════════════════════════════════════════════════════════
# EHRBASE AQL CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class EHRbaseClient:
    def __init__(self, base_url, user, password):
        self.base = base_url.rstrip("/") + "/ehrbase/rest/openehr/v1"
        self.auth = HTTPBasicAuth(user, password)
        self.h = {"Content-Type": "application/json", "Accept": "application/json"}

    def aql(self, query: str, params: dict = None) -> list[dict]:
        body = {"q": query}
        if params:
            body["query_parameters"] = params
        r = requests.post(f"{self.base}/query/aql",
                          auth=self.auth, headers=self.h, json=body, timeout=30)
        if r.status_code == 200:
            data = r.json()
            cols = [c["name"] for c in data.get("columns", [])]
            return [dict(zip(cols, row)) for row in data.get("rows", [])]
        print(f"  ⚠️  AQL {r.status_code}: {r.text[:150]}")
        return []

    def all_ehr_ids(self) -> list[str]:
        return [r["ehr_id"] for r in self.aql("SELECT e/ehr_id/value AS ehr_id FROM EHR e")]


# ══════════════════════════════════════════════════════════════════════════════
# AQL DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════

class DataFetcher:
    def __init__(self, client: EHRbaseClient):
        self.c = client

    def conditions(self, ehr_id):
        # Fetch condition names — single clean query, handle nulls in Python
        rows = self.c.aql(
            "SELECT eval/data[at0001]/items[at0002]/value/value AS name "
            "FROM EHR e CONTAINS COMPOSITION c "
            "CONTAINS EVALUATION eval[openEHR-EHR-EVALUATION.problem_diagnosis.v1] "
            "WHERE e/ehr_id/value = $ehr_id",
            {"ehr_id": ehr_id})
        return [r["name"] for r in rows if r.get("name")]

    def family_history(self, ehr_id):
        rows = self.c.aql(
            "SELECT eval/data[at0001]/items[at0028]/items[at0029]/value/value AS fh "
            "FROM EHR e CONTAINS COMPOSITION c "
            "CONTAINS EVALUATION eval[openEHR-EHR-EVALUATION.family_history.v2] "
            "WHERE e/ehr_id/value = $ehr_id",
            {"ehr_id": ehr_id})
        return [r["fh"] for r in rows if r.get("fh")]

    def latest_bp(self, ehr_id):
        rows = self.c.aql(
            "SELECT obs/data[at0001]/events[at0006]/data[at0003]/items[at0004]/value/magnitude AS sys, "
            "obs/data[at0001]/events[at0006]/data[at0003]/items[at0005]/value/magnitude AS dia, "
            "obs/data[at0001]/events[at0006]/time/value AS time "
            "FROM EHR e CONTAINS COMPOSITION c "
            "CONTAINS OBSERVATION obs[openEHR-EHR-OBSERVATION.blood_pressure.v2] "
            "WHERE e/ehr_id/value = $ehr_id "
            "ORDER BY obs/data[at0001]/events[at0006]/time/value DESC LIMIT 1",
            {"ehr_id": ehr_id})
        return rows[0] if rows else {}

    def latest_weight(self, ehr_id):
        rows = self.c.aql(
            "SELECT obs/data[at0002]/events[at0003]/data[at0001]/items[at0004]/value/magnitude AS weight, "
            "obs/data[at0002]/events[at0003]/time/value AS time "
            "FROM EHR e CONTAINS COMPOSITION c "
            "CONTAINS OBSERVATION obs[openEHR-EHR-OBSERVATION.body_weight.v2] "
            "WHERE e/ehr_id/value = $ehr_id "
            "ORDER BY obs/data[at0002]/events[at0003]/time/value DESC LIMIT 1",
            {"ehr_id": ehr_id})
        return rows[0].get("weight") if rows else None

    def latest_bmi(self, ehr_id):
        rows = self.c.aql(
            "SELECT obs/data[at0001]/events[at0002]/data[at0003]/items[at0004]/value/magnitude AS bmi, "
            "obs/data[at0001]/events[at0002]/time/value AS time "
            "FROM EHR e CONTAINS COMPOSITION c "
            "CONTAINS OBSERVATION obs[openEHR-EHR-OBSERVATION.body_mass_index.v2] "
            "WHERE e/ehr_id/value = $ehr_id "
            "ORDER BY obs/data[at0001]/events[at0002]/time/value DESC LIMIT 1",
            {"ehr_id": ehr_id})
        return rows[0].get("bmi") if rows else None

    def smoking(self, ehr_id):
        rows = self.c.aql(
            "SELECT eval/data[at0001]/items[at0089]/value/value AS status "
            "FROM EHR e CONTAINS COMPOSITION c "
            "CONTAINS EVALUATION eval[openEHR-EHR-EVALUATION.tobacco_smoking_summary.v2] "
            "WHERE e/ehr_id/value = $ehr_id LIMIT 1",
            {"ehr_id": ehr_id})
        return rows[0].get("status") if rows else None

    def labs(self, ehr_id) -> dict[str, float]:
        """Return latest numeric value per LOINC code, extracted from Conclusion field."""
        rows = self.c.aql(
            "SELECT "
            "obs/data[at0001]/events[at0002]/data[at0003]/items[at0005]/value/defining_code/code_string AS loinc, "
            "obs/data[at0001]/events[at0002]/data[at0003]/items[at0057]/value/value AS conclusion, "
            "obs/data[at0001]/events[at0002]/time/value AS time "
            "FROM EHR e CONTAINS COMPOSITION c "
            "CONTAINS OBSERVATION obs[openEHR-EHR-OBSERVATION.laboratory_test_result.v1] "
            "WHERE e/ehr_id/value = $ehr_id "
            "ORDER BY obs/data[at0001]/events[at0002]/time/value DESC",
            {"ehr_id": ehr_id})

        # Latest per LOINC, extract numeric from conclusion string
        seen = {}
        for r in rows:
            loinc = r.get("loinc")
            if loinc and loinc not in seen:
                val = _extract_numeric(r.get("conclusion", ""))
                if val is not None:
                    seen[loinc] = val
        return seen


def _extract_numeric(conclusion: str) -> float | None:
    if not conclusion:
        return None
    m = re.search(r':\s*([\d.]+)', conclusion)
    if m:
        try: return float(m.group(1))
        except ValueError: pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PROFILE BUILDER
# Maps EHRbase data onto the exact profile field names used in the rule strings.
# ══════════════════════════════════════════════════════════════════════════════

# LOINC → profile field (must match variable names in rule expressions)
LOINC_TO_FIELD = {
    "2093-3":  "patient_last_blood_totalcholesterol",
    "2089-1":  "patient_last_blood_ldlcholesterol",
    "2085-9":  "patient_last_blood_hdlcholesterol",
    "2571-8":  "patient_last_blood_triglycerides",
    "89579-7": "patient_last_blood_troponint",
    "6598-7":  "patient_last_blood_troponini",
    "4548-4":  "patient_last_blood_hba1c",
    "2339-0":  "patient_last_blood_glucose",
    "14118-4": "patient_last_vitals_lactate",
    "30934-4": "patient_last_vitals_bnp",
    "2160-0":  "patient_last_vitals_creatinine",
    "33763-4": "patient_last_vitals_ntprobnp",
    "718-7":   "patient_last_vitals_haemoglobin",
    "62238-1": "patient_last_vitals_egfr",
    "13969-1": "patient_last_vitals_ckmb",
    "1884-6":  "patient_last_vitals_apob",
}

# Condition name → profile boolean field
CONDITION_FLAGS = {
    "patient_last_condition_diabetes":       ["diabetes"],
    "patient_last_condition_chd":            ["coronary artery disease", "coronary heart disease", "cad"],
    "patient_last_condition_hypertension":   ["hypertension"],
    "patient_last_condition_hyperlipidemia": ["hyperlipid", "hypercholesterol"],
    "patient_last_condition_angina":         ["angina"],
    "patient_last_condition_atherosclerosis":["atherosclerosis"],
}

# Symptom name → profile field (symptoms stored as conditions)
SYMPTOM_FLAGS = {
    "patient_last_symptom_fatigue":       ["fatigue"],
    "patient_last_symptom_chestpain":     ["chest pain"],
    "patient_last_symptom_breathlessness":["breathlessness", "shortness of breath"],
    "patient_last_symptom_dizziness":     ["dizziness"],
    "patient_last_symptom_gastrodiscomfort":["gastrointestinal discomfort", "gastro"],
    "patient_last_symptom_weakness":      ["weakness"],
    "patient_last_symptom_confusion":     ["confusion"],
}

CARDIAC_FH_KEYWORDS = [
    "angina", "coronary", "ischaemic heart", "atherosclerosis",
    "heart disease", "heart failure", "cardiogenic", "myocardial infarction",
    "cardiac", "hypertension"
]


def _any_match(texts: list[str], keywords: list[str]) -> bool:
    return any(any(kw in t.lower() for kw in keywords) for t in texts)


def build_profile(ehr_id: str, fetcher: DataFetcher,
                  dob: str = None, gender: str = None,
                  ethnicity: str = None) -> dict:

    p: dict[str, Any] = {
        "patient_id":              ehr_id,
        "profile_generated_at":    datetime.utcnow().isoformat(),
        "patient_ethnicity":       ethnicity or "unknown",
    }

    # ── Age & gender ──────────────────────────────────────────────────────────
    if dob:
        try:
            today = date.today()
            d = date.fromisoformat(dob)
            age = today.year - d.year - ((today.month, today.day) < (d.month, d.day))
            p["patient_age"] = age
        except Exception:
            p["patient_age"] = 0
    else:
        p["patient_age"] = 0

    # Rule strings use 'male'/'female' lowercase
    p["patient_gender"] = (gender or "unknown").lower()

    # ── Vitals ────────────────────────────────────────────────────────────────
    bp = fetcher.latest_bp(ehr_id)
    p["patient_last_vitals_systolicbp"]  = bp.get("sys")
    p["patient_last_vitals_diastolicbp"] = bp.get("dia")
    p["patient_weight"] = fetcher.latest_weight(ehr_id)
    p["patient_imc"]    = fetcher.latest_bmi(ehr_id)

    # ── Smoking & alcohol ─────────────────────────────────────────────────────
    smk = fetcher.smoking(ehr_id)
    smoking_active = smk not in (None, "Never smoked", "Former smoker")
    # Rule strings compare: patient_last_smoking == 'true'
    p["patient_last_smoking"] = "true" if smoking_active else "false"
    p["patient_last_alcohol"]  = "false"  # not yet captured in EHRbase

    # ── Lab results ───────────────────────────────────────────────────────────
    labs = fetcher.labs(ehr_id)
    for loinc, field in LOINC_TO_FIELD.items():
        p[field] = labs.get(loinc, 0)  # default 0 so numeric comparisons work

    # ── Conditions ────────────────────────────────────────────────────────────
    conditions = fetcher.conditions(ehr_id)

    for field, keywords in CONDITION_FLAGS.items():
        p[field] = "true" if _any_match(conditions, keywords) else "false"

    # ── Symptoms (stored as conditions in current template) ───────────────────
    for field, keywords in SYMPTOM_FLAGS.items():
        p[field] = "true" if _any_match(conditions, keywords) else "false"

    # ── Family history ────────────────────────────────────────────────────────
    fh = fetcher.family_history(ehr_id)
    p["patient_last_famhistory"] = "true" if _any_match(fh, CARDIAC_FH_KEYWORDS) else "false"

    return p


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT METADATA
# DOB and gender until demographic archetype is wired up
# ══════════════════════════════════════════════════════════════════════════════

STATIC_META = {
    # Case study patients (fixed template IDs)
    "cccbac30-4548-8b40-b209-bfe386af2383": {"dob": "1977-03-15", "gender": "male"},
    "7eadd38b-2cdb-726a-b8f0-83446f70c757": {"dob": "1958-06-20", "gender": "female"},
    "b38b712b-744d-68e5-f9eb-b88133542f01": {"dob": "1957-04-10", "gender": "male"},
    "fddf394d-8240-4a47-bb6e-7c6c7f20ec73": {"dob": "1947-02-28", "gender": "female"},
}

DUMMY_NAME_META = {
    "Arjun Sharma":     {"dob": "1967-01-01", "gender": "male"},
    "Priya Patel":      {"dob": "1981-01-01", "gender": "female"},
    "James O'Brien":    {"dob": "1954-01-01", "gender": "male"},
    "Mei-Lin Chen":     {"dob": "1992-01-01", "gender": "female"},
    "Fatima Al-Hassan": {"dob": "1965-01-01", "gender": "female"},
}


def load_meta() -> dict:
    meta = dict(STATIC_META)
    # Search broadly for the map files — works regardless of working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    # Build list of candidate base directories
    search_dirs = set([
        cwd,
        os.path.join(cwd, ".."),
        script_dir,
        os.path.join(script_dir, ".."),
        os.path.expanduser("~/Desktop/mystera-ehrbase"),
    ])
    for map_name in ["dummy-data/ehr_id_map.json", "dummy-data/case_study_ehr_map.json"]:
        for base in search_dirs:
            map_file = os.path.normpath(os.path.join(base, map_name))
            if os.path.exists(map_file):
                with open(map_file) as f:
                    ehr_map = json.load(f)
                for name, ehr_id in ehr_map.items():
                    if ehr_id and ehr_id not in meta:
                        meta[ehr_id] = DUMMY_NAME_META.get(name, {"dob": None, "gender": "unknown"})
                break
    return meta


# ══════════════════════════════════════════════════════════════════════════════
# AGENT-FACING PROFILE
# Converts a run_patient() result into the exact shape MysteraCardiacEnv expects
# (see env/cardiac_env.py and data/sample_profiles.py SAMPLE_PROFILES): a flat
# dict of the fields the env reads, plus a `rule_labels` dict keyed by the
# env's 3-disease cardiac-agent scope.
# ══════════════════════════════════════════════════════════════════════════════

# scoring.py's DISEASE_NAMES -> env.cardiac_env.DISEASES. Renames only where the
# two engines' naming conventions differ (coronaryarterydisease -> cad); every
# disease the rules engine scores is passed through to the agent.
AGENT_DISEASE_KEYS = {
    "angina":                 "angina",
    "atherosclerosis":        "atherosclerosis",
    "cardiogenicshock":       "cardiogenicshock",
    "coronaryarterydisease":  "cad",
}


def _as_bool(value: Any) -> bool:
    return str(value).lower() == "true"


def to_agent_profile(result: dict) -> dict:
    p = result["profile"]
    risk = result["risk_profile"]
    return {
        "patient_id":                        result["ehr_id"],
        "patient_imc":                        p.get("patient_imc") or 0,
        "patient_last_smoking":               _as_bool(p.get("patient_last_smoking")),
        "patient_last_famhistory":            _as_bool(p.get("patient_last_famhistory")),
        "patient_last_blood_troponint":       p.get("patient_last_blood_troponint", 0),
        "patient_last_vitals_systolicbp":     p.get("patient_last_vitals_systolicbp") or 0,
        "patient_last_blood_ldlcholesterol":  p.get("patient_last_blood_ldlcholesterol", 0),
        "patient_last_blood_hdlcholesterol":  p.get("patient_last_blood_hdlcholesterol", 0),
        "patient_last_ecg_ischemia":          False,  # not yet captured in EHRbase
        "rule_labels": {
            agent_key: risk[score_key].lower().replace(" ", "_")
            for score_key, agent_key in AGENT_DISEASE_KEYS.items()
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE — per patient
# ══════════════════════════════════════════════════════════════════════════════

def run_patient(ehr_id: str, fetcher: DataFetcher, meta: dict) -> dict:
    m = meta.get(ehr_id, {})
    profile      = build_profile(ehr_id, fetcher, dob=m.get("dob"), gender=m.get("gender"))
    rule_results = evaluate_all_rules(profile)
    risk         = classify_risk(rule_results)
    triggered    = [k for k, v in rule_results.items() if v]
    result = {
        "ehr_id":          ehr_id,
        "profile":         profile,
        "rules_triggered": triggered,
        "risk_profile":    risk,
    }
    result["agent_profile"] = to_agent_profile(result)
    return result


DEBUG = False  # set via --debug flag

def print_result(result: dict):
    p   = result["profile"]
    pid = result["ehr_id"][:8]
    age = p.get("patient_age", "?")
    gen = p.get("patient_gender", "?")
    if DEBUG:
        print(f"  [DEBUG] fatigue={p.get('patient_last_symptom_fatigue')} "
              f"chestpain={p.get('patient_last_symptom_chestpain')} "
              f"tropI={p.get('patient_last_blood_troponini')} "
              f"tropT={p.get('patient_last_blood_troponint')} "
              f"famhist={p.get('patient_last_famhistory')}")

    print(f"\n  ┌─ {pid}...  Age: {age}  Gender: {gen}")

    has_risk = False
    for disease, level in result["risk_profile"].items():
        if level == RISK_CONFIRMED:
            print(f"  │  🔴 {disease:<26} → {level}")
            has_risk = True
        elif level == RISK_UNCONFIRMED:
            print(f"  │  🟡 {disease:<26} → {level}")
            has_risk = True
    if not has_risk:
        print(f"  │  ⚪ No diseases flagged")

    # Show triggered rules (exclude No Risk noise)
    triggered = [r for r in result["rules_triggered"] if "confirmed" not in r]
    if triggered:
        print(f"  │  Rules: {', '.join(triggered)}")

    # Key lab/vital values
    vals = []
    for f, lbl in [
        ("patient_last_blood_troponini", "TropI"),
        ("patient_last_blood_troponint", "TropT"),
        ("patient_last_vitals_creatinine", "Creat"),
        ("patient_last_vitals_bnp", "BNP"),
        ("patient_last_blood_ldlcholesterol", "LDL"),
        ("patient_last_blood_triglycerides", "TG"),
        ("patient_last_vitals_apob", "ApoB"),
        ("patient_last_vitals_systolicbp", "SBP"),
        ("patient_imc", "BMI"),
    ]:
        v = p.get(f)
        if v:
            vals.append(f"{lbl}={v}")
    if vals:
        print(f"  │  {' | '.join(vals)}")
    print(f"  └{'─'*62}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MyStera ProfileIQ Rules Scoring")
    parser.add_argument("--ehrbase-url", default=DEFAULT_URL)
    parser.add_argument("--user",        default=DEFAULT_USER)
    parser.add_argument("--password",    default=DEFAULT_PASSWORD)
    parser.add_argument("--patient-id",  help="Score a single EHR ID")
    parser.add_argument("--output",      help="Save results to JSON file")
    parser.add_argument("--agent-output", help="Save patient profiles in MysteraCardiacEnv-compatible "
                                                "format (same list-of-dicts shape as SAMPLE_PROFILES "
                                                "in data/sample_profiles.py)")
    parser.add_argument("--debug",       action="store_true", help="Print raw profile values")
    args = parser.parse_args()
    global DEBUG
    DEBUG = args.debug

    client  = EHRbaseClient(args.ehrbase_url, args.user, args.password)
    fetcher = DataFetcher(client)
    meta    = load_meta()

    print("\n═══════════════════════════════════════════════════════")
    print("  MyStera ProfileIQ — Rules Scoring Engine")
    print("  4 rule groups: non-mod | mod | symptoms | blood markers")
    print("═══════════════════════════════════════════════════════")

    ehr_ids = [args.patient_id] if args.patient_id else client.all_ehr_ids()
    print(f"\n  Scoring {len(ehr_ids)} patient(s)...\n")

    results = []
    counters = {RISK_CONFIRMED: 0, RISK_UNCONFIRMED: 0, RISK_NONE: 0}

    for ehr_id in ehr_ids:
        result = run_patient(ehr_id, fetcher, meta)
        results.append(result)
        print_result(result)
        for level in result["risk_profile"].values():
            counters[level] = counters.get(level, 0) + 1

    print(f"\n═══ Summary ════════════════════════════════════════════")
    print(f"  Patients:          {len(results)}")
    print(f"  Confirmed risks:   {counters.get(RISK_CONFIRMED, 0)}")
    print(f"  Unconfirmed risks: {counters.get(RISK_UNCONFIRMED, 0)}")
    print(f"═══════════════════════════════════════════════════════\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results saved → {args.output}\n")

    if args.agent_output:
        with open(args.agent_output, "w") as f:
            json.dump([r["agent_profile"] for r in results], f, indent=2, default=str)
        print(f"  Agent-compatible profiles saved → {args.agent_output}\n")


if __name__ == "__main__":
    main()
"""
Synthetic doctor-correction generator.
--------------------------------------
There is no real doctor-correction data yet (no review UI, no feedback table),
so this stands in for it during POC training: it derives a "latent severity"
score per disease from the same 8 raw features the agent observes, deliberately
using different logic than the blunt OR/AND rule engine (see
data/synthetic_patients.py's `generic_rule_label`) so there is a learnable gap
between what the rules flag and what a clinician would conclude — that gap is
exactly what the agent is supposed to learn to close. A small amount of
label noise is added to represent ordinary inter-rater variability.

Weights are illustrative, not clinically validated — good enough to give the
POC a learnable, non-arbitrary signal, not a diagnostic claim.
"""

import numpy as np

from env.cardiac_env import DISEASES, LABEL_NAMES, normalise

# Per-disease feature weights (each sums to 1). "hdl" is protective, so its
# contribution is inverted ((1 - normalised hdl)) wherever it appears.
DISEASE_WEIGHTS = {
    "angina": {
        "smoking": 0.20, "famhistory": 0.15, "troponin": 0.30,
        "sbp": 0.15, "bmi": 0.10, "ecg": 0.10,
    },
    "atherosclerosis": {
        "ldl": 0.30, "hdl_inv": 0.25, "bmi": 0.15,
        "smoking": 0.15, "sbp": 0.15,
    },
    "cardiogenicshock": {
        "troponin": 0.35, "ecg": 0.25, "sbp": 0.20,
        "bmi": 0.10, "smoking": 0.10,
    },
    "cad": {
        "troponin": 0.25, "sbp": 0.20, "ldl": 0.15,
        "smoking": 0.20, "famhistory": 0.10, "ecg": 0.10,
    },
}

# Bucket cutoffs for the weighted severity score (asymmetric on purpose —
# confirmed risk should be rarer than no/unconfirmed in a realistic population).
NO_RISK_CUTOFF, CONFIRMED_CUTOFF = 0.35, 0.60

# Probability the simulated doctor's bucket shifts by one level from the
# "true" severity bucket, representing ordinary inter-rater noise.
LABEL_NOISE_P = 0.12


def _feature_values(profile):
    return {
        "bmi": normalise(profile["patient_imc"], "patient_imc"),
        "smoking": 1.0 if profile["patient_last_smoking"] else 0.0,
        "famhistory": 1.0 if profile["patient_last_famhistory"] else 0.0,
        "troponin": normalise(profile["patient_last_blood_troponint"], "patient_last_blood_troponint"),
        "sbp": normalise(profile["patient_last_vitals_systolicbp"], "patient_last_vitals_systolicbp"),
        "ldl": normalise(profile["patient_last_blood_ldlcholesterol"], "patient_last_blood_ldlcholesterol"),
        "hdl_inv": 1.0 - normalise(profile["patient_last_blood_hdlcholesterol"], "patient_last_blood_hdlcholesterol"),
        "ecg": 1.0 if profile["patient_last_ecg_ischemia"] else 0.0,
    }


def severity_score(profile, disease):
    values = _feature_values(profile)
    weights = DISEASE_WEIGHTS[disease]
    return sum(weights[feat] * values[feat] for feat in weights)


def _bucket(score):
    if score < NO_RISK_CUTOFF:
        return 0
    if score < CONFIRMED_CUTOFF:
        return 1
    return 2


def simulate_doctor_labels(profile, rng=None):
    """Return a {disease: label} dict standing in for a doctor's ground-truth correction."""
    rng = rng if rng is not None else np.random.default_rng()
    labels = {}
    for disease in DISEASES:
        level = _bucket(severity_score(profile, disease))
        if rng.random() < LABEL_NOISE_P:
            level += rng.choice([-1, 1])
            level = int(np.clip(level, 0, len(LABEL_NAMES) - 1))
        labels[disease] = LABEL_NAMES[level]
    return labels

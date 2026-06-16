"""
Synthetic patient generator for training-scale data.
------------------------------------------------------
SAMPLE_PROFILES (5 hand-written patients) is far too small to train a policy
on. This generates as many synthetic patients as needed by sampling
clinically-plausible raw feature distributions, deriving `rule_labels` from
a simplified, self-contained approximation of scoring.py's blunt OR/AND rule
style (kept independent of scoring.py so this needs no EHRbase connection),
and pairing each with a simulated doctor correction (see doctor_simulator.py).

generic_rule_label uses the same thresholds scoring.py's real rules use
(troponin T > 0.01 ng/mL, LDL > 160 mg/dL) where the equivalent field exists
in the agent's 8-field schema; non-mod/mod rules collapse to a single
"lifestyle" flag (smoking or family history) since age/ethnicity aren't part
of that schema. It is deliberately blunter than doctor_simulator's weighted
severity score — that gap is what the agent has to learn to close.
"""

import numpy as np

from env.cardiac_env import DISEASES
from data.doctor_simulator import simulate_doctor_labels


def _sample_raw_features(rng):
    return {
        "patient_imc": float(np.clip(rng.normal(27, 5), 15, 45)),
        "patient_last_smoking": bool(rng.random() < 0.30),
        "patient_last_famhistory": bool(rng.random() < 0.25),
        "patient_last_blood_troponint": float(np.clip(rng.exponential(0.015), 0, 0.1)),
        "patient_last_vitals_systolicbp": float(np.clip(rng.normal(130, 20), 90, 200)),
        "patient_last_blood_ldlcholesterol": float(np.clip(rng.normal(120, 40), 50, 250)),
        "patient_last_blood_hdlcholesterol": float(np.clip(rng.normal(50, 15), 20, 100)),
        "patient_last_ecg_ischemia": bool(rng.random() < 0.15),
    }


def generic_rule_label(raw, disease):
    troponin_fired = raw["patient_last_blood_troponint"] > 0.01
    lifestyle_fired = raw["patient_last_smoking"] or raw["patient_last_famhistory"]

    if disease == "atherosclerosis":
        blood_fired = raw["patient_last_blood_ldlcholesterol"] > 160
    else:
        blood_fired = troponin_fired

    if lifestyle_fired and blood_fired:
        return "confirmed_risk"
    if lifestyle_fired or blood_fired:
        return "unconfirmed_risk"
    return "no_risk"


def generate_synthetic_patients(n, seed=None, prefix="SYN"):
    """Returns (profiles, doctor_labels), each a list of length n."""
    rng = np.random.default_rng(seed)
    profiles, doctor_labels = [], []
    for i in range(n):
        raw = _sample_raw_features(rng)
        profile = {
            "patient_id": f"{prefix}{i:05d}",
            **raw,
            "rule_labels": {d: generic_rule_label(raw, d) for d in DISEASES},
        }
        profiles.append(profile)
        doctor_labels.append(simulate_doctor_labels(profile, rng))
    return profiles, doctor_labels

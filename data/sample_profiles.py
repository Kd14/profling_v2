"""
Synthetic cardiac-domain patient profiles for manual episode simulation
(see RL_agent.py). Field names follow the Mystera profile schema; each
profile carries the generic, rule-based interim risk labels for all four
in-scope cardiac diseases (angina, atherosclerosis, cardiogenicshock, cad).
DOCTOR_LABELS holds the corresponding ground-truth correction the doctor
would make for that patient — the signal MysteraCardiacEnv rewards the
agent against.

Patients are constructed to cover the range of corrections the agent must
learn: P001 needs an angina UPGRADE, P002 and P004 are already correct
(KEEP across all four diseases), P003 needs both a cad DOWNGRADE and a
cardiogenicshock DOWNGRADE (rules over-called risk twice for this patient),
and P005 needs both an atherosclerosis UPGRADE and a cardiogenicshock
UPGRADE — illustrating that a single action per episode can't fix every
disease a patient needs corrected in one step.
"""

SAMPLE_PROFILES = [
    {
        "patient_id": "P001",
        "patient_imc": 31.2,
        "patient_last_smoking": True,
        "patient_last_famhistory": True,
        "patient_last_blood_troponint": 0.018,
        "patient_last_vitals_systolicbp": 148,
        "patient_last_blood_ldlcholesterol": 172,
        "patient_last_blood_hdlcholesterol": 38,
        "patient_last_ecg_ischemia": False,
        "rule_labels": {
            "angina": "unconfirmed_risk",
            "atherosclerosis": "no_risk",
            "cardiogenicshock": "no_risk",
            "cad": "no_risk",
        },
    },
    {
        "patient_id": "P002",
        "patient_imc": 23.5,
        "patient_last_smoking": False,
        "patient_last_famhistory": False,
        "patient_last_blood_troponint": 0.003,
        "patient_last_vitals_systolicbp": 118,
        "patient_last_blood_ldlcholesterol": 95,
        "patient_last_blood_hdlcholesterol": 58,
        "patient_last_ecg_ischemia": False,
        "rule_labels": {
            "angina": "no_risk",
            "atherosclerosis": "no_risk",
            "cardiogenicshock": "no_risk",
            "cad": "no_risk",
        },
    },
    {
        "patient_id": "P003",
        "patient_imc": 27.8,
        "patient_last_smoking": False,
        "patient_last_famhistory": True,
        "patient_last_blood_troponint": 0.006,
        "patient_last_vitals_systolicbp": 132,
        "patient_last_blood_ldlcholesterol": 110,
        "patient_last_blood_hdlcholesterol": 50,
        "patient_last_ecg_ischemia": False,
        "rule_labels": {
            "angina": "unconfirmed_risk",
            "atherosclerosis": "unconfirmed_risk",
            "cardiogenicshock": "unconfirmed_risk",
            "cad": "confirmed_risk",
        },
    },
    {
        "patient_id": "P004",
        "patient_imc": 34.6,
        "patient_last_smoking": True,
        "patient_last_famhistory": True,
        "patient_last_blood_troponint": 0.062,
        "patient_last_vitals_systolicbp": 168,
        "patient_last_blood_ldlcholesterol": 205,
        "patient_last_blood_hdlcholesterol": 32,
        "patient_last_ecg_ischemia": True,
        "rule_labels": {
            "angina": "confirmed_risk",
            "atherosclerosis": "confirmed_risk",
            "cardiogenicshock": "confirmed_risk",
            "cad": "confirmed_risk",
        },
    },
    {
        "patient_id": "P005",
        "patient_imc": 29.1,
        "patient_last_smoking": False,
        "patient_last_famhistory": False,
        "patient_last_blood_troponint": 0.009,
        "patient_last_vitals_systolicbp": 126,
        "patient_last_blood_ldlcholesterol": 158,
        "patient_last_blood_hdlcholesterol": 44,
        "patient_last_ecg_ischemia": False,
        "rule_labels": {
            "angina": "no_risk",
            "atherosclerosis": "no_risk",
            "cardiogenicshock": "no_risk",
            "cad": "unconfirmed_risk",
        },
    },
]

DOCTOR_LABELS = [
    {"angina": "confirmed_risk", "atherosclerosis": "no_risk", "cardiogenicshock": "no_risk", "cad": "no_risk"},
    {"angina": "no_risk", "atherosclerosis": "no_risk", "cardiogenicshock": "no_risk", "cad": "no_risk"},
    {"angina": "unconfirmed_risk", "atherosclerosis": "unconfirmed_risk", "cardiogenicshock": "no_risk", "cad": "unconfirmed_risk"},
    {"angina": "confirmed_risk", "atherosclerosis": "confirmed_risk", "cardiogenicshock": "confirmed_risk", "cad": "confirmed_risk"},
    {"angina": "no_risk", "atherosclerosis": "unconfirmed_risk", "cardiogenicshock": "unconfirmed_risk", "cad": "unconfirmed_risk"},
]

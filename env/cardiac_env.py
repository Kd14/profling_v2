"""
MysteraCardiacEnv
-----------------
Gymnasium environment for the cardiac domain RL agent described in the
Mystera architecture: each episode samples one patient's generic,
rule-based interim risk profile; the agent chooses a (disease, correction)
pair to adjust one disease label, and is rewarded by how close the result
lands to the doctor's eventual correction.

Cold-start gate: a patient with fewer than COLD_START_THRESHOLD prior
interactions gets the generic profile passed through unchanged regardless
of the chosen action — the agent is silenced until it has enough history
for that patient.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

DISEASES = ["angina", "atherosclerosis", "cardiogenicshock", "cad"]
DISEASE_DISPLAY_NAMES = {
    "angina": "Angina",
    "atherosclerosis": "Atherosclerosis",
    "cardiogenicshock": "Cardiogenic Shock",
    "cad": "CAD",
}
LABEL_NAMES = ["no_risk", "unconfirmed_risk", "confirmed_risk"]
LABEL_MAP = {name: level for level, name in enumerate(LABEL_NAMES)}

KEEP, UPGRADE, DOWNGRADE = 0, 1, 2

COLD_START_THRESHOLD = 3

# Clinically reasonable min/max used to normalise raw profile fields into [0, 1]
NORM_RANGES = {
    "patient_imc": (15.0, 45.0),
    "patient_last_blood_troponint": (0.0, 0.1),
    "patient_last_vitals_systolicbp": (90.0, 200.0),
    "patient_last_blood_ldlcholesterol": (50.0, 250.0),
    "patient_last_blood_hdlcholesterol": (20.0, 100.0),
}


def normalise(value, field):
    lo, hi = NORM_RANGES[field]
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def bool_to_float(value):
    return 1.0 if value else 0.0


def build_observation(profile):
    """Map a raw profile dict to the 11-dim observation vector the agent sees."""
    rule_labels = profile["rule_labels"]
    return np.array(
        [
            normalise(profile["patient_imc"], "patient_imc"),
            bool_to_float(profile["patient_last_smoking"]),
            bool_to_float(profile["patient_last_famhistory"]),
            normalise(profile["patient_last_blood_troponint"], "patient_last_blood_troponint"),
            normalise(profile["patient_last_vitals_systolicbp"], "patient_last_vitals_systolicbp"),
            normalise(profile["patient_last_blood_ldlcholesterol"], "patient_last_blood_ldlcholesterol"),
            normalise(profile["patient_last_blood_hdlcholesterol"], "patient_last_blood_hdlcholesterol"),
            bool_to_float(profile["patient_last_ecg_ischemia"]),
            *[LABEL_MAP[rule_labels[d]] / 2.0 for d in DISEASES],
        ],
        dtype=np.float32,
    )


def _apply_correction(label, correction):
    level = LABEL_MAP[label]
    if correction == UPGRADE:
        level = min(level + 1, len(LABEL_NAMES) - 1)
    elif correction == DOWNGRADE:
        level = max(level - 1, 0)
    return LABEL_NAMES[level]


class MysteraCardiacEnv(gym.Env):
    """One decision per episode: pick (disease, correction) for a sampled patient."""

    metadata = {"render_modes": []}

    def __init__(self, profiles, doctor_labels, patient_history=None):
        super().__init__()
        if len(profiles) != len(doctor_labels):
            raise ValueError("profiles and doctor_labels must be the same length")

        self.profiles = profiles
        self.doctor_labels = doctor_labels
        self.patient_history = patient_history if patient_history is not None else {}

        self.action_space = spaces.Discrete(len(DISEASES) * 3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8 + len(DISEASES),), dtype=np.float32
        )

        self._current_idx = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current_idx = int(self.np_random.integers(0, len(self.profiles)))
        profile = self.profiles[self._current_idx]
        obs = build_observation(profile)
        info = {"patient_id": profile["patient_id"]}
        return obs, info

    def step(self, action):
        if self._current_idx is None:
            raise RuntimeError("Call reset() before step()")

        disease_idx = int(action) // 3
        correction = int(action) % 3
        disease = DISEASES[disease_idx]

        profile = self.profiles[self._current_idx]
        patient_id = profile["patient_id"]
        doctor = self.doctor_labels[self._current_idx]

        prior_interactions = self.patient_history.get(patient_id, 0)
        cold_start_active = prior_interactions < COLD_START_THRESHOLD

        final_labels = dict(profile["rule_labels"])
        if not cold_start_active:
            final_labels[disease] = _apply_correction(final_labels[disease], correction)

        # Graded reward: +1 exact match, 0 off-by-one, -1 off-by-two — rewards
        # the agent for moving in the right direction even short of an exact hit.
        reward_breakdown = {}
        for d in DISEASES:
            distance = abs(LABEL_MAP[final_labels[d]] - LABEL_MAP[doctor[d]])
            reward_breakdown[d] = float(1.0 - distance)

        reward = float(np.mean(list(reward_breakdown.values())))

        self.patient_history[patient_id] = prior_interactions + 1

        obs = build_observation(profile)
        terminated = True
        truncated = False
        info = {
            "patient_id": patient_id,
            "final_labels": final_labels,
            "reward_breakdown": reward_breakdown,
            "cold_start_active": cold_start_active,
        }
        return obs, reward, terminated, truncated, info

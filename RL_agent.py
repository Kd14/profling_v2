"""

Manual Episode Simulation
--------------------------
Walk through a single episode step-by-step to understand what
the agent is doing and why. Useful for debugging and building
intuition before running full training.

Run with:
  python notebooks/simulate_episode.py

No training required — uses a random (untrained) policy to show
the environment mechanics clearly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env.cardiac_env import (
    MysteraCardiacEnv, DISEASES, DISEASE_DISPLAY_NAMES, LABEL_MAP, LABEL_NAMES,
    KEEP, UPGRADE, DOWNGRADE, COLD_START_THRESHOLD
)
from data.sample_profiles import SAMPLE_PROFILES, DOCTOR_LABELS


def decode_action(action_int):
    """Convert action integer to human-readable string."""
    disease_idx = action_int // 3
    correction = action_int % 3
    disease = DISEASES[disease_idx]
    corr_name = {KEEP: "KEEP", UPGRADE: "UPGRADE", DOWNGRADE: "DOWNGRADE"}[correction]
    return disease, corr_name


def simulate_manual(patient_idx=0, override_action=None, bypass_coldstart=True):
    """
    Run one episode for a specific patient and print every detail.

    Parameters
    ----------
    patient_idx : int
        Index into SAMPLE_PROFILES (0–4)
    override_action : int or None
        Force a specific action (0–(len(DISEASES)*3-1)) instead of random
    bypass_coldstart : bool
        If True, give the patient 10 prior interactions so the agent acts
    """
    patient_id = SAMPLE_PROFILES[patient_idx]["patient_id"]
    history = {patient_id: 10} if bypass_coldstart else {}

    env = MysteraCardiacEnv(
        profiles=SAMPLE_PROFILES,
        doctor_labels=DOCTOR_LABELS,
        patient_history=history,
    )

    # Force the environment to the chosen patient by cycling reset
    obs, info = env.reset()
    while env._current_idx != patient_idx:
        obs, info = env.reset()

    print("\n" + "█"*60)
    print("  EPISODE SIMULATION")
    print("█"*60)

    # Show the raw observation vector with labels. The fixed clinical features
    # come first, then one "Current label" entry per disease in DISEASES — kept
    # in sync automatically as the agent's scope expands to more diseases.
    feature_names = [
        "BMI (normalised)",
        "Smoking",
        "Family history",
        "Troponin T (normalised)",
        "Systolic BP (normalised)",
        "LDL cholesterol (normalised)",
        "HDL cholesterol (normalised)",
        "ECG ischemia",
    ] + [f"Current label: {DISEASE_DISPLAY_NAMES[d]}" for d in DISEASES]
    print(f"\n  Patient: {patient_id}")
    print(f"\n  Observation vector (what the agent sees):")
    for name, val in zip(feature_names, obs):
        bar = "█" * int(val * 20)
        print(f"    {name:<35} {val:.3f}  {bar}")

    # Show generic rule labels
    generic = SAMPLE_PROFILES[patient_idx]["rule_labels"]
    print(f"\n  Generic rule labels (from SQL rules engine):")
    for d, label in generic.items():
        print(f"    {d:<18} → {label.upper()}")

    # Choose action
    if override_action is not None:
        action = override_action
        print(f"\n  Action (manual override): {action}")
    else:
        action = env.action_space.sample()
        print(f"\n  Action (random sample): {action}")

    disease, corr_name = decode_action(action)
    print(f"  Decoded: act on [{disease}] → {corr_name}")

    if not bypass_coldstart:
        print(f"\n  ⚠ Cold-start active — agent has < {COLD_START_THRESHOLD} "
              f"interactions. Action will be suppressed (pass-through).")

    # Step
    obs_new, reward, terminated, truncated, info = env.step(action)

    # Show results
    doctor = DOCTOR_LABELS[patient_idx]
    print(f"\n  Results:")
    print(f"  {'Disease':<18} {'Generic':^14} {'Agent':^14} {'Doctor':^14} {'Reward':^8}")
    print(f"  {'-'*68}")

    for d in DISEASES:
        g = generic[d]
        a = info["final_labels"][d]
        doc = doctor[d]
        r = info["reward_breakdown"][d]
        tick = "✓" if a == doc else "✗"
        print(f"  {d:<18} {g:^14} {a:^14} {doc:^14} {r:^6.1f}  {tick}")

    print(f"\n  Total episode reward: {reward:.3f}  (mean across diseases)")
    print(f"  Cold-start active:    {info['cold_start_active']}")
    print("\n" + "█"*60 + "\n")
    return reward, info


def run_all_patients():
    """Simulate one episode for every sample patient, random actions."""
    print("\n" + "="*60)
    print("  ALL PATIENTS — RANDOM POLICY BASELINE")
    print("  (shows environment behaviour before any training)")
    print("="*60)

    rewards = []
    for i, profile in enumerate(SAMPLE_PROFILES):
        r, _ = simulate_manual(patient_idx=i, bypass_coldstart=True)
        rewards.append(r)

    print(f"\n  Mean reward across all patients (random policy): {np.mean(rewards):.3f}")
    print(f"  (A trained agent should score significantly higher)\n")


if __name__ == "__main__":
    # Show detailed walkthrough for patient P001
    print("  Detailed walkthrough — Patient P001")
    print("  Action 1 = upgrade Angina (disease 0, correction 1)")
    simulate_manual(patient_idx=0, override_action=1, bypass_coldstart=True)

    # Then run all patients with random policy to show baseline
    run_all_patients()
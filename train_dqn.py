"""
Train a DQN policy for MysteraCardiacEnv on synthetic data, and evaluate it
against a random-policy baseline. There is no real doctor-correction data
yet (see data/doctor_simulator.py) — this proves the learning loop actually
works end to end, not a clinically validated model.

DQN fits this environment well: every episode is exactly one decision (see
MysteraCardiacEnv.step — terminated=True after the first step), so this is
effectively a contextual bandit with a 12-dim discrete action space and a
12-dim continuous observation; off-policy replay with a small MLP is a
natural, sample-efficient match.

Cold-start is bypassed for both training and evaluation (patient_history is
pre-seeded past COLD_START_THRESHOLD) so every step carries a real, action-
dependent reward — the gate itself was already covered by a separate smoke
test and isn't what this script is trying to validate.

Run with:
  python3 train_dqn.py
"""

import os

import numpy as np
from stable_baselines3 import DQN

from env.cardiac_env import MysteraCardiacEnv, DISEASES, COLD_START_THRESHOLD, build_observation
from data.synthetic_patients import generate_synthetic_patients

TRAIN_N = 2000
EVAL_N = 400
TOTAL_TIMESTEPS = 60_000
SEED = 0


def make_env(profiles, doctor_labels):
    # Pre-seed history past the cold-start threshold so every step is
    # action-dependent — see module docstring.
    history = {p["patient_id"]: COLD_START_THRESHOLD for p in profiles}
    return MysteraCardiacEnv(profiles=profiles, doctor_labels=doctor_labels, patient_history=history)


def evaluate(env, model, profiles, doctor_labels):
    """Deterministic pass over every profile in order, so the random and
    trained policies are scored on the exact same patients (paired comparison)."""
    env.reset(seed=SEED)
    rewards = []
    rule_correct, post_correct = 0, 0

    for idx in range(len(profiles)):
        env._current_idx = idx
        profile = profiles[idx]
        doctor = doctor_labels[idx]
        obs = build_observation(profile)

        if model is None:
            action = int(env.action_space.sample())
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])

        _, reward, _, _, step_info = env.step(action)
        rewards.append(reward)

        disease = DISEASES[action // 3]
        rule_correct += int(profile["rule_labels"][disease] == doctor[disease])
        post_correct += int(step_info["final_labels"][disease] == doctor[disease])

    n = len(profiles)
    return {
        "mean_reward": float(np.mean(rewards)),
        "rule_accuracy_on_acted_disease": rule_correct / n,
        "post_action_accuracy_on_acted_disease": post_correct / n,
    }


def main():
    train_profiles, train_doctor = generate_synthetic_patients(TRAIN_N, seed=SEED)
    eval_profiles, eval_doctor = generate_synthetic_patients(EVAL_N, seed=SEED + 1)  # held out, different seed

    train_env = make_env(train_profiles, train_doctor)

    print(f"Training DQN for {TOTAL_TIMESTEPS} timesteps on {TRAIN_N} synthetic patients...")
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=5e-4,
        buffer_size=50_000,
        learning_starts=1000,
        target_update_interval=1000,
        exploration_fraction=0.2,
        verbose=0,
        seed=SEED,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)

    print("Evaluating random vs trained policy on held-out synthetic patients (paired, same patients)...\n")

    eval_env = make_env(eval_profiles, eval_doctor)
    eval_env.action_space.seed(SEED)
    random_stats = evaluate(eval_env, None, eval_profiles, eval_doctor)
    trained_stats = evaluate(eval_env, model, eval_profiles, eval_doctor)

    print(f"{'Metric':<42} {'Random':>10} {'Trained':>10}")
    print("-" * 64)
    for key in ["mean_reward", "rule_accuracy_on_acted_disease", "post_action_accuracy_on_acted_disease"]:
        print(f"{key:<42} {random_stats[key]:>10.3f} {trained_stats[key]:>10.3f}")

    os.makedirs("models", exist_ok=True)
    model.save("models/dqn_cardiac")
    print("\nSaved model -> models/dqn_cardiac.zip")


if __name__ == "__main__":
    main()

# MyStera Cardiac Domain RL System — Technical Documentation

**Scope:** the full pipeline from raw openEHR clinical data through rule-based risk
scoring to the reinforcement-learning agent that personalises those risk labels over
time. Covers two repositories: `~/Desktop/mystera-ehrbase` (the openEHR/EHRbase backend
and dummy clinical data) and `~/profling_v2` (this repo — the rules engine, RL
environment, and training pipeline).

**Status as of this writing:** the rules engine and EHRbase integration are live and
working against real (dummy) patient data. The RL agent and its training loop are a
working proof-of-concept trained on **synthetic** doctor-correction data, because no
real doctor-feedback data exists yet. This is called out explicitly everywhere it's
relevant — nothing in this document should be read as a clinical validation claim.

---

# Part I — High-Level Summary

## 1. What this system does

A rules engine produces a generic, population-level cardiac risk classification for a
patient from their EHR data. A reinforcement-learning agent sits downstream of that
rules engine and learns to *personalise* those classifications, using a doctor's
corrections as its training signal. Over time, the agent should converge on
recommendations that need less correction than the raw rules output would.

```
EHRbase (openEHR)
   │  AQL queries
   ▼
Profiling Layer  (scoring.py: EHRbaseClient, DataFetcher, build_profile)
   │  flat profile dict
   ▼
Rules Engine  (scoring.py: evaluate_all_rules, classify_risk)
   │  Interim Risk Profile  (Confirmed / Unconfirmed / No Risk per disease)
   ▼
Agent-Profile Adapter  (scoring.py: to_agent_profile)
   │  schema-normalised profile + rule_labels
   ▼
Cardiac Domain RL Agent  (env/cardiac_env.py: MysteraCardiacEnv + trained policy)
   │  Finalised Risk Profile  (one disease label corrected per decision)
   ▼
Doctor review  (today: simulated — data/doctor_simulator.py)
   │  correction signal
   ▼
Training / feedback loop  (train_dqn.py)
   └──> updates the policy, which improves the next round of agent decisions
```

## 2. Layer-by-layer: inputs, transforms, outputs

### Layer 0 — EHRbase (openEHR clinical data store)

| | |
|---|---|
| **Input** | Clinical data authored against two openEHR *templates*: `MyStera_Onboarding_v1` (demographics, lifestyle, family history) and `MyStera_ClinicalEncounter_v1` (conditions, medications, allergies, observations, diagnostic reports). Loaded via `scripts/setup_ehrbase.py` and `scripts/load_dummy_data.py` in the `mystera-ehrbase` repo. |
| **Transform** | EHRbase persists each patient's data as a versioned set of *compositions* (structured clinical documents) inside an *EHR* (one per patient), validated against the uploaded template definitions, and stored in PostgreSQL. |
| **Output** | A queryable openEHR data store, accessed via AQL (Archetype Query Language — openEHR's SQL-like query language for clinical data) over HTTP. Currently running with 9 patients: 5 "dummy" patients with general clinical profiles, and 4 cardiac case-study patients purpose-built for specific disease scenarios (Angina ×2, Atherosclerosis ×1, Cardiogenic Shock ×1). |

### Layer 1 — Profiling (`scoring.py`: `EHRbaseClient`, `DataFetcher`, `build_profile`)

| | |
|---|---|
| **Input** | An EHR ID. Six AQL queries per patient: conditions, family history, latest blood pressure, latest weight, latest BMI, smoking status, and latest lab results (by LOINC code). |
| **Transform** | Each AQL result is mapped onto a specific profile field name (e.g. LOINC `2089-1` → `patient_last_blood_ldlcholesterol`). Free-text condition/symptom names are keyword-matched into boolean flags (e.g. any condition containing "angina" → `patient_last_condition_angina = "true"`). Age is derived from date of birth (sourced from a static lookup table, since there's no demographics archetype query wired up yet). |
| **Output** | A flat **profile dict** — the system's canonical per-patient representation, keyed by field names that exactly match the variable names used in the rule expressions (Layer 2). |

### Layer 2 — Rules Engine (`scoring.py`: `evaluate_all_rules`, `classify_risk`)

| | |
|---|---|
| **Input** | The Layer 1 profile dict. |
| **Transform** | ~17 boolean rule expressions (age/sex/ethnicity thresholds, smoking/alcohol flags, symptom flags, lab-value thresholds, and 4 composite "confirmed disease" rules) are evaluated against the profile by a custom expression parser. Each disease's classification is then: **Confirmed Risk** if its composite rule fires, **Unconfirmed Risk** if any subsidiary rule fires, else **No Risk**. |
| **Output** | The **Interim Risk Profile**: `{angina: ..., atherosclerosis: ..., cardiogenicshock: ..., coronaryarterydisease: ...}`, each one of `"Confirmed Risk"` / `"Unconfirmed Risk"` / `"No Risk"`. This is the generic, population-level classification — identical logic for every patient, no personalisation yet. |

### Layer 3 — Agent-Profile Adapter (`scoring.py`: `to_agent_profile`)

| | |
|---|---|
| **Input** | The Layer 2 result (profile + Interim Risk Profile). |
| **Transform** | Reconciles two independently-evolved schemas: string booleans (`"true"`/`"false"`) become real `bool`s; risk-label strings are lowercased/underscored (`"Confirmed Risk"` → `"confirmed_risk"`); disease keys are renamed where the two systems disagree (`coronaryarterydisease` → `cad`); fields the agent doesn't need are dropped; one field the EHRbase side never captures (`patient_last_ecg_ischemia`) is defaulted. |
| **Output** | An **agent profile** — the exact shape `MysteraCardiacEnv` consumes: 8 raw clinical fields + a `rule_labels` dict. This is the hand-off point between "rules engine" and "RL agent," and the two could be deployed/scaled independently of each other from here. |

### Layer 4 — Cardiac Domain RL Agent (`env/cardiac_env.py`: `MysteraCardiacEnv` + a trained policy)

| | |
|---|---|
| **Input** | An agent profile (Layer 3) + that patient's interaction count (`patient_history`, for the cold-start gate). |
| **Transform** | The profile is normalised into an 12-dimensional observation vector. The policy chooses one of 12 actions: (disease ∈ {angina, atherosclerosis, cardiogenicshock, cad}) × (correction ∈ {KEEP, UPGRADE, DOWNGRADE}). The chosen correction is applied to *that one disease's* label; the other three pass through unchanged. If the patient has fewer than 3 prior interactions, the action is ignored entirely and the generic label passes through (cold-start gate). |
| **Output** | The **Finalised Risk Profile** — the same four-disease dict as Layer 2, but with (at most) one label adjusted based on what the policy has learned. |

### Layer 5 — Doctor review (today: simulated)

| | |
|---|---|
| **Input** | The Finalised Risk Profile. |
| **Transform** | **In production, this is a human clinician** reviewing the case and accepting or correcting each disease label — that correction is the real reward signal the whole system is built around. **No such review UI or feedback table exists yet.** As a stand-in, `data/doctor_simulator.py` computes a synthetic "ground truth" from a different (weighted, continuous) function of the same clinical features, deliberately diverging from the rules engine's blunt boolean logic, plus a small amount of random label noise to mimic inter-rater variation. |
| **Output** | A `{disease: label}` correction — real, eventually; simulated, today. |

### Layer 6 — Training / feedback loop (`train_dqn.py`)

| | |
|---|---|
| **Input** | (observation, action, reward) tuples generated by running the policy (or a random policy, early on) against many patients and comparing its decisions to the Layer 5 correction. |
| **Transform** | A DQN (Deep Q-Network) reinforcement learning algorithm updates the policy's weights via experience replay, to make higher-reward actions more likely in similar future states. |
| **Output** | An updated policy checkpoint (`models/dqn_cardiac.zip`), which becomes the policy used in Layer 4 going forward. **First POC training run:** mean reward improved from 0.297 (random policy) to 0.509 (trained), and accuracy on the disease the policy chose to act on improved from 38.5% to 77.0%, against held-out synthetic data — proof the loop functions, not a clinical result. |

## 3. What's real vs. simulated vs. not yet built, at a glance

| Component | Status |
|---|---|
| EHRbase + openEHR data | **Real** — live Docker deployment, 9 patients loaded |
| Profiling (Layer 1) | **Real** — runs against live EHRbase data today |
| Rules engine (Layer 2) | **Real** — pure logic, no external dependency |
| Agent-profile adapter (Layer 3) | **Real** |
| RL environment (Layer 4, mechanics) | **Real** — gymnasium-compliant, unit-tested |
| RL policy (Layer 4, the learned weights) | **POC** — trained only on synthetic data |
| Doctor review (Layer 5) | **Simulated** — no real UI/data exists |
| Training loop (Layer 6) | **Real mechanism, synthetic data** |
| `patient_history` persistence | **Not built** — in-memory only, lost on restart |
| Production orchestrator (online/offline pipelines, model registry, monitoring) | **Not built** — see §11 for the target design |

---

# Part II — Detailed Technical Reference

## 4. Architecture & design philosophy

**Why frame this as personalisation-via-correction rather than a prediction model?**
The rules engine already encodes a reasonable, explainable population-level baseline.
Replacing it outright with a learned model would throw away that explainability and
require enormous amounts of labelled data to match its baseline competence. Instead,
the agent's job is narrower and more tractable: learn the *systematic ways this
specific patient's case differs from what the rules engine assumes* — and only act
when it has enough history with that patient to do so responsibly (the cold-start
gate, §7.3).

**Why one agent per clinical domain, not one per disease?**
Cardiac biomarkers (troponin, ECG, cholesterol, blood pressure) are simultaneously
relevant to angina, atherosclerosis, cardiogenic shock, and CAD. Splitting into four
separate disease-level agents would mean four separate models all consuming the same
features in isolation, unable to share what they learn about, say, "this patient's
troponin readings tend to run high for unrelated reasons." A single domain-level agent
sees the whole feature set at once and outputs a per-disease decision from it.

**Why cardiogenic shock is in scope** despite the original framing only naming
Angina/Atherosclerosis/CAD: the rules engine already scores it (it's one of the four
`DISEASE_NAMES` in `scoring.py`), and one of the four live EHRbase case-study patients
(`CS3-CardiogenicShock-Female`) exists specifically to exercise it. Excluding a disease
the upstream engine already classifies — and that test data already targets — would be
an arbitrary narrowing with no basis in the data. The system's stated direction is to
keep expanding scope to more diseases/domains over time, not to prune it.

## 5. Layer 0 deep dive — EHRbase & the openEHR data model

### 5.1 What openEHR/EHRbase is, and why it was chosen

openEHR is an open clinical-data specification designed so that the *meaning* of
clinical data is defined independently of any particular database schema. Two
concepts matter for this system:

- **Archetypes** — reusable, vendor-neutral definitions of a clinical concept (e.g.
  "blood pressure observation," "problem/diagnosis evaluation"), each with a stable
  identifier like `openEHR-EHR-OBSERVATION.blood_pressure.v2`. These are the actual
  query targets in Layer 1's AQL.
- **Templates** — a constrained, application-specific combination of archetypes (e.g.
  "everything captured during a clinical encounter"). MyStera defines two:
  `MyStera_Onboarding_v1` and `MyStera_ClinicalEncounter_v1` (`.opt.xml` files in
  `mystera-ehrbase/templates/`).

**EHRbase** is the open-source openEHR server implementation used here — it stores
data per the openEHR information model (backed by PostgreSQL) and exposes AQL over a
REST API. The appeal for a system like this is that the clinical data model is decoupled
from the application: new clinical concepts (templates, archetypes) can be added
without an application-side schema migration, and the data is queryable with a
standard, vendor-neutral query language (AQL) rather than bespoke table joins.

### 5.2 Deployment

`mystera-ehrbase/docker-compose.yml` defines two services:

| Service | Image | Port | Role |
|---|---|---|---|
| `ehrdb` | `ehrbase/ehrbase-v2-postgres:16.2` | 5432 | PostgreSQL backing store |
| `ehrbase` | `ehrbase/ehrbase:2.31.0` | 8080 | openEHR REST API server |

Auth is HTTP Basic (`SECURITY_AUTHTYPE: BASIC`), with separate regular (`mystera`) and
admin (`mystemadmin`) accounts — `scoring.py`'s `EHRbaseClient` authenticates as the
regular user. The AQL endpoint is `POST /ehrbase/rest/openehr/v1/query/aql`.

**Setup sequence** (from `mystera-ehrbase/README.md`): `docker compose up -d` →
`scripts/setup_ehrbase.py` (waits for EHRbase, uploads both templates, smoke-tests with
a test EHR) → `scripts/load_dummy_data.py` (loads 5 dummy patients) →
`scripts/case_study.py` (loads 4 cardiac case-study patients, based on the script
names found in the repo).

### 5.3 Current data

| Patient set | Count | Notes |
|---|---|---|
| Dummy patients (`dummy-data/ehr_id_map.json`) | 5 | Arjun Sharma (58M, hypertension/T2DM), Priya Patel (45F, anxiety/penicillin allergy), James O'Brien (72M, CAD/heart failure/AF), Mei-Lin Chen (34F, healthy baseline), Fatima Al-Hassan (61F, CKD stage 3a/anaemia) |
| Case-study patients (`dummy-data/case_study_ehr_map.json`) | 4 | `CS1-Angina-Male`, `CS1-Angina-Female`, `CS2-Atherosclerosis-Male`, `CS3-CardiogenicShock-Female` — purpose-built per disease scenario. **No CAD-specific case study exists** — worth creating one given CAD is the fourth in-scope disease. |

### 5.4 Known gap worth flagging

A direct query to the live instance's template-list endpoint
(`/ehrbase/rest/openehr/v1/definition/template/adl1.4`) returned only
`MyStera_ClinicalEncounter_v1` — `MyStera_Onboarding_v1` did not show up, despite the
`.opt.xml` file existing in `templates/` and the README describing both as uploaded.
Family-history data still resolved correctly in a live test query, so this isn't
silently breaking anything observed so far, but it should be verified directly (re-run
`scripts/setup_ehrbase.py`'s upload step, or check EHRbase logs) before relying on any
onboarding-archetype field being populated.

## 6. Layer 1 deep dive — Profiling (`scoring.py`)

### 6.1 AQL query design

Each `DataFetcher` method is a single AQL query scoped to one EHR ID, generally of the
shape:

```sql
SELECT obs/data[...]/events[...]/data[...]/items[...]/value/magnitude AS field
FROM EHR e CONTAINS COMPOSITION c
CONTAINS OBSERVATION obs[openEHR-EHR-OBSERVATION.<archetype>.v<n>]
WHERE e/ehr_id/value = $ehr_id
ORDER BY ... DESC LIMIT 1
```

The `at0001`/`at0002`-style path segments are openEHR **archetype node IDs** — stable
identifiers for a specific data point within an archetype's internal structure (e.g.
"systolic pressure" within the blood-pressure observation archetype). `ORDER BY ... DESC
LIMIT 1` implements "most recent value" directly in AQL rather than fetching everything
and filtering client-side.

### 6.2 Lab results and LOINC

`labs()` fetches all `laboratory_test_result` observations for a patient, then maps
each by its LOINC code (a standardised vocabulary for lab tests) onto a profile field
via `LOINC_TO_FIELD`:

| LOINC | Profile field |
|---|---|
| 2093-3 | `patient_last_blood_totalcholesterol` |
| 2089-1 | `patient_last_blood_ldlcholesterol` |
| 2085-9 | `patient_last_blood_hdlcholesterol` |
| 2571-8 | `patient_last_blood_triglycerides` |
| 89579-7 | `patient_last_blood_troponint` |
| 6598-7 | `patient_last_blood_troponini` |
| 4548-4 | `patient_last_blood_hba1c` |
| 2339-0 | `patient_last_blood_glucose` |
| 14118-4 | `patient_last_vitals_lactate` |
| 30934-4 | `patient_last_vitals_bnp` |
| 2160-0 | `patient_last_vitals_creatinine` |
| 33763-4 | `patient_last_vitals_ntprobnp` |
| 718-7 | `patient_last_vitals_haemoglobin` |
| 62238-1 | `patient_last_vitals_egfr` |
| 13969-1 | `patient_last_vitals_ckmb` |
| 1884-6 | `patient_last_vitals_apob` |

The actual numeric value is extracted from a free-text "conclusion" field via regex
(`_extract_numeric`) rather than a structured numeric field — a pragmatic choice given
the current template stores lab results that way, but a fragile one (any conclusion
text not matching `: <number>` silently yields no value).

### 6.3 Conditions, symptoms, and family history → booleans

Conditions and symptoms are both stored as free-text condition names in the current
template and reduced to booleans via keyword matching (`CONDITION_FLAGS`,
`SYMPTOM_FLAGS`, case-insensitive substring match). Family history is similarly
keyword-matched against `CARDIAC_FH_KEYWORDS` (angina, coronary, heart disease/failure,
cardiogenic, MI, hypertension, etc.) — if *any* family-history entry mentions a cardiac
keyword, `patient_last_famhistory = "true"`.

This is a real limitation: keyword matching against free text is brittle (a condition
named "Suspected angina — ruled out" would still match) and structurally provisional —
once conditions/symptoms get their own dedicated boolean fields in the openEHR
template, this should be replaced with a direct field read.

### 6.4 Known gaps in this layer

- `patient_last_alcohol` is hardcoded `"false"` — never captured from EHRbase.
- Age/gender/ethnicity come from a hardcoded Python dict (`STATIC_META` +
  `DUMMY_NAME_META`), not from EHRbase at all — there's no demographics archetype query
  implemented yet, despite `MyStera_Onboarding_v1` presumably covering this.
- BMI and cholesterol fields returned `0` for a real patient tested mid-build (see
  conversation history) — either not loaded for that patient or a path mismatch; not
  yet root-caused.

## 7. Layer 2 deep dive — Rules Engine

### 7.1 Why a custom expression parser instead of `eval()`

Rules are stored as strings (`rules_non_mod`, `rules_mod`, etc. in `scoring.py`) so
they can eventually live in a database/XML rather than Python source. Evaluating
arbitrary strings with Python's built-in `eval()` against that kind of externally-
editable input is a classic remote-code-execution risk — a rule string containing
`__import__('os').system(...)` would execute arbitrary code. `scoring.py` instead
implements a small **recursive-descent parser** (`_tokenise` → `_Parser`) that only
understands a deliberately narrow grammar:

```
expr     := or_expr
or_expr  := and_expr ('|' and_expr)*
and_expr := cmp_expr ('&' cmp_expr)*
cmp_expr := '(' expr ')' | value op value
op       := '==' | '!=' | '>' | '<' | '>=' | '<='
```

This can only ever compare a profile field, a number, or a string literal — there is no
way to express anything beyond boolean logic over comparisons, which closes off the
code-execution risk entirely regardless of where the rule strings come from.
`&amp;`/`&gt;`/`&lt;` decoding (`decode_rule`) exists so the same rule strings survive
round-tripping through XML/HTML-escaped storage.

### 7.2 Rule categories and the four-disease taxonomy

`DISEASE_NAMES = ["angina", "atherosclerosis", "cardiogenicshock", "coronaryarterydisease"]`.
For each disease, rules fall into:

| Category | Example | Purpose |
|---|---|---|
| Non-modifiable | `rule_non_mod_angina_age`: male>45 or female>55 | Fixed risk factors (age, sex, ethnicity) |
| Modifiable | `rule_mod_angina`: smoking or alcohol | Lifestyle risk factors |
| Symptoms | `rule_symptoms_angina`: fatigue | Self-reported/observed symptoms |
| Blood markers | `rule_blood_angina`: troponin I>0.04 or troponin T>0.01 or creatinine>5 | Objective lab evidence |
| Composite (confirmed) | `rule_confirmed_angina`: (non-mod OR ethnicity OR mod) AND symptom AND blood | The actual classifier for "Confirmed Risk" |

`classify_risk` then applies, per disease: **Confirmed Risk** if the composite rule
fires; else **Unconfirmed Risk** if *any* subsidiary rule (`UNCONFIRMED_RULES`) fires;
else **No Risk**. This is a deliberately conservative design — "confirmed" requires
corroborating evidence across categories (a risk factor *and* a symptom *and* a lab
abnormality), while "unconfirmed" flags anything with at least one signal, erring
toward sensitivity over specificity for the lighter-weight label.

## 8. Layer 3 deep dive — Agent-Profile Adapter

`scoring.py` and `env/cardiac_env.py` were built independently and use different
conventions (string vs. bool, `"Confirmed Risk"` vs. `"confirmed_risk"`,
`coronaryarterydisease` vs. `cad`). `to_agent_profile()` is the single reconciliation
point — deliberately the *only* place this translation happens, so the two systems can
otherwise evolve independently. `AGENT_DISEASE_KEYS` is an explicit map (not a generic
string-transform) specifically so a future schema divergence fails loudly (`KeyError`)
rather than silently producing a wrong key.

```python
AGENT_DISEASE_KEYS = {
    "angina":                "angina",
    "atherosclerosis":       "atherosclerosis",
    "cardiogenicshock":      "cardiogenicshock",
    "coronaryarterydisease": "cad",
}
```

`run_patient()` embeds the result as `result["agent_profile"]`, and
`scoring.py --agent-output FILE` writes `[r["agent_profile"] for r in results]` — a
file directly loadable as `MysteraCardiacEnv(profiles=json.load(open(FILE)), ...)`.

## 9. Layer 4 deep dive — the RL Environment

### 9.1 Why Gymnasium

[Gymnasium](https://gymnasium.farama.org/) (the maintained successor to OpenAI Gym) is
the de facto standard interface for RL environments in Python: `reset() -> (obs,
info)`, `step(action) -> (obs, reward, terminated, truncated, info)`, plus
`action_space`/`observation_space` declarations. Building `MysteraCardiacEnv` against
this interface — rather than a bespoke one — means it works out of the box with any
Gymnasium-compatible training library (Stable-Baselines3, used here; RLlib; CleanRL;
etc.) without writing any adapter code. It costs nothing (it's a small, dependency-light
library, no account/API key/network access required) and buys broad tooling
compatibility for free.

### 9.2 MDP formulation

| Element | Definition |
|---|---|
| **State / observation** | 12-dim float vector in [0,1]: BMI, smoking, family history, troponin T, systolic BP, LDL, HDL, ECG ischemia (8 normalised clinical features) + one normalised "current label" per disease (4 diseases, value ∈ {0, 0.5, 1} for no/unconfirmed/confirmed). |
| **Action** | `Discrete(12)` = 4 diseases × 3 corrections. `action // 3` selects the disease, `action % 3` selects KEEP(0)/UPGRADE(1)/DOWNGRADE(2). |
| **Transition** | Deterministic given the action: the chosen disease's label moves one level up/down (clamped at the boundaries) or stays; the other three diseases' labels are untouched. |
| **Reward** | Graded by label-distance to the doctor's correction, summed per disease then averaged: `+1` exact match, `0` off-by-one, `-1` off-by-two, for *every* disease (not just the one acted on) — see §9.4. |
| **Episode length** | Always 1 step (`terminated=True` immediately after `step()`). Each "episode" is one decision for one patient. |

### 9.3 Why episodes are length-1 (and why that matters for algorithm choice)

Each patient interaction is treated as a single, independent decision rather than a
multi-step sequence — there's no notion of "the agent's action now changes what
happens to this patient next." This makes the problem, formally, a **contextual
bandit**: pick the best action given the current context (observation), get an
immediate reward, repeat with a new, independent context. This directly motivated the
algorithm choice in §10.1.

### 9.4 Why a graded reward instead of binary match/mismatch

A binary reward (`+1` correct / `-1` wrong) gives the same penalty to "off by one risk
level" and "off by two risk levels" — but those are not equally bad decisions
clinically (recommending "unconfirmed" for a patient who is actually "confirmed" is a
much smaller error than recommending "no risk"). The graded reward
(`1 - |level(final) - level(doctor)|`) gives partial credit for directionally-correct
corrections and a harsher penalty for being maximally wrong, which is both a better
training signal (smoother gradient, less reward sparsity) and a better match to
clinical reality.

### 9.5 The cold-start gate

```python
COLD_START_THRESHOLD = 3
...
cold_start_active = prior_interactions < COLD_START_THRESHOLD
final_labels = dict(profile["rule_labels"])
if not cold_start_active:
    final_labels[disease] = _apply_correction(final_labels[disease], correction)
```

For a patient's first 3 interactions, **whatever action the policy chooses is
discarded** — the generic rule-engine label is what gets shown, every time. This is a
deliberate safety design: a freshly-deployed (or freshly-encountered-patient) policy
has no track record for that patient, and the system would rather fall back to the
well-understood, explainable rules-engine baseline than let an under-informed policy
adjust a real clinical label. `patient_history` (a `{patient_id: interaction_count}`
dict) tracks this — currently **in-memory only**, which means it resets on every
process restart; this needs to move to durable storage before any real deployment (see
§11).

Note that *reward is still computed* during cold-start (comparing the pass-through
label to the doctor's correction) — but since the final label doesn't depend on the
chosen action during cold-start, that reward carries no information about which action
was best. `train_dqn.py` deliberately pre-seeds `patient_history` past the threshold
during training/eval for exactly this reason (see §10.3).

## 10. Layer 4b / 6 deep dive — Policy, Training, and the Synthetic Data Problem

### 10.1 Why DQN

Given the contextual-bandit framing (§9.3), the natural family of algorithms is
**off-policy, value-based** methods — DQN (Deep Q-Network) being the standard choice
for a discrete action space of this size (12 actions) with a small, fully-observed,
continuous state (12 floats). DQN's experience replay buffer is a particularly good
fit here: because every transition is independent (no temporal credit assignment
needed across steps), the network can be trained on randomly-shuffled past
transitions with no loss of validity, which is exactly what a replay buffer does.
On-policy alternatives like PPO are built for problems with long-horizon credit
assignment (where *when* you collect a rollout matters) — machinery this problem
doesn't need.

Implementation: [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (SB3),
the most widely used, well-tested Gymnasium-native RL library — chosen over writing a
DQN from scratch because the algorithm itself (replay buffer, target network,
ε-greedy exploration schedule, Huber loss) is well-understood, easy to get subtly
wrong, and not the part of this system that's actually novel.

### 10.2 The synthetic-data problem this had to solve first

There is no real doctor-correction data: no review UI, no feedback table, nothing.
Without *some* reward signal, there's nothing to train against. Two new modules exist
purely to unblock this, and both are explicitly meant to be retired once real data
exists:

**`data/synthetic_patients.py`** — generates patients at training scale (the 5
hand-written `SAMPLE_PROFILES` are far too few to train on) by sampling clinically
plausible distributions for each of the 8 raw fields (e.g. BMI ~ Normal(27, 5) clipped
to [15,45]; troponin T ~ Exponential(0.015) clipped to [0, 0.1] — right-skewed, since
real troponin values are mostly low with an elevated tail). `rule_labels` are derived
from `generic_rule_label()`, a simplified, self-contained re-implementation of
`scoring.py`'s blunt OR/AND rule logic (same real thresholds where the field exists —
troponin T > 0.01, LDL > 160 — collapsed lifestyle gate since age/ethnicity aren't part
of the 8-field agent schema). It's self-contained (no EHRbase dependency) so training
data can be generated instantly, offline, at any volume.

**`data/doctor_simulator.py`** — stands in for the doctor's correction. Computes a
per-disease **weighted severity score** over the same 8 features
(`DISEASE_WEIGHTS`, e.g. angina weights troponin 0.30, smoking 0.20, family history
0.15, systolic BP 0.15, BMI 0.10, ECG 0.10), thresholds it into the three risk buckets
(`NO_RISK_CUTOFF=0.35`, `CONFIRMED_CUTOFF=0.60`), and adds 12% random one-level label
noise to mimic ordinary inter-rater variability. Critically, this uses **different
logic** than the rule engine — a different function of the same features, not a noisy
copy of `generic_rule_label`. That's intentional: if the "doctor" were just the rule
engine plus noise, there would be no learnable structure for the agent to find. By
using an independent function, there's a genuine, non-arbitrary gap between "what the
rules flag" and "what the doctor would conclude," and closing that gap is exactly the
agent's job. On 2,000 generated patients, rule-vs-doctor agreement came out to 32-58%
per disease — meaningful daylight, not noise.

Weights and cutoffs in both modules are **illustrative, not clinically validated** —
good enough to produce a non-arbitrary, learnable signal for a POC, not a basis for any
clinical claim.

### 10.3 Training setup (`train_dqn.py`)

| Parameter | Value | Why |
|---|---|---|
| Training patients | 2,000 synthetic | Large enough for the DQN to see substantial coverage of the feature space |
| Eval patients | 400 synthetic, different RNG seed | Genuinely held out — not seen during training |
| Total timesteps | 60,000 | ~30 visits/patient on average; ample for an 8-feature→12-action mapping with a small MLP |
| Learning rate | 5e-4 | SB3 default (1e-4) tuned up slightly — small dense network, no need for the conservative default used for larger image-based DQN setups |
| Buffer size | 50,000 | Covers most of a training run's transitions |
| `learning_starts` | 1,000 | Steps of pure random exploration before training begins |
| `target_update_interval` | 1,000 | Lowered from SB3's default (10,000) so the target network actually updates a few times within this run's modest step budget |
| `exploration_fraction` | 0.2 | ε anneals over the first 20% of training |
| Cold-start handling | Bypassed (patient_history pre-seeded to `COLD_START_THRESHOLD`) | During cold-start, reward is action-independent (§9.5) — training on those steps would inject pure noise into the gradient. The gate mechanism itself already has a dedicated smoke test; this script isn't trying to re-validate it. |

### 10.4 Evaluation methodology

`evaluate()` runs **both** the random-policy baseline and the trained policy over the
exact same 400 held-out patients, in the same order — a paired comparison, so the
reported gap reflects the policy's quality rather than which patients happened to be
sampled. For each patient it records: the env's graded reward, whether the rule label
already matched the doctor's label on the disease the policy chose to act on (baseline
"do nothing" accuracy), and whether the label matched *after* the action (post-action
accuracy).

### 10.5 First results

| Metric | Random policy | Trained DQN |
|---|---|---|
| Mean reward (range -1 to 1) | 0.297 | **0.509** |
| Accuracy on acted-upon disease, pre-action (rule vs. doctor) | 42.5%* | 20.0%* |
| Accuracy on acted-upon disease, post-action | 38.5% | **77.0%** |

*\*These differ between columns because each policy chooses a different disease to act
on per patient — the trained policy systematically seeks out diseases where the rule
label diverges most from the doctor's, which is exactly where intervening has the most
value, even though the "before" accuracy on those self-selected diseases is lower.*

The headline number is the post-action accuracy jump (38.5% → 77.0%) and the reward
jump (0.297 → 0.509): the agent is reliably choosing actions that move the label
toward the (synthetic) doctor's ground truth far more often than chance, confirming
the full loop — environment, reward, replay, gradient updates — works correctly.

## 11. Production Roadmap — what's missing between this POC and a deployable system

This section formalises the orchestrator design discussed during development. Two
pipelines, different cadences, sharing one data layer:

```
ONLINE  (per patient, low latency)
  EHR event/composition committed
    -> rules engine (Layer 2)
    -> agent-profile adapter (Layer 3)
    -> inference service (loads current policy checkpoint, reads patient_history)
    -> Finalised Risk Profile written to a DB table
    -> doctor's UI reads it, doctor submits correction back to the same table

OFFLINE  (scheduled, batch)
  accumulate corrections since last training run
    -> retrain/fine-tune policy
    -> evaluate against held-out data AND the currently-deployed model (regression check)
    -> human approval gate
    -> promote new model version
    -> inference service picks up the new version
```

### 11.1 Components still needed, in priority order

1. **Patient state store.** `patient_history` must move from an in-memory dict to a
   real table (e.g. Postgres `patient_agent_state(patient_id, interaction_count, ...)`)
   — the most direct gap between this POC and anything resembling production.
2. **Feedback capture table.** For every patient/disease/encounter: `rule_label`,
   `agent_recommendation`, `doctor_final_label`, timestamp, doctor ID. This is the
   schema that will eventually replace `doctor_simulator.py`/`synthetic_patients.py`
   entirely — designing it well now matters more than any other single piece of this
   roadmap.
3. **Inference service.** A thin API (e.g. FastAPI `/score` endpoint) loading the
   model once, stateless on weights, reading/writing patient state from #1.
4. **Model registry.** Versioned checkpoints + metadata (training data snapshot, eval
   metrics, timestamp). Doesn't need to be MLflow on day one — a `models/` folder plus
   a metadata row per version in Postgres is enough until scale demands more.
5. **Retraining job with a human approval gate, not auto-deploy.** In a
   clinical-adjacent system, a policy that silently updates itself in a fully closed
   loop is a genuine safety and auditability problem, not just a nice-to-have control —
   you need to be able to answer "which model version made this recommendation, and
   why" months later.
6. **Rollback.** Trivial once #4 is versioned — point the inference service back at
   the previous checkpoint.
7. **Drift monitoring.** Track the distribution of agent actions and accuracy over
   time; a sudden shift (e.g. a spike in UPGRADE recommendations) is an early signal of
   either real population drift or a bug, ideally caught before it becomes a
   patient-safety issue.

### 11.2 Sizing this to the project's actual stage

Don't reach for Airflow/Kafka/MLflow yet. A Postgres table for state + feedback, one
FastAPI service with `/score` and `/correct` endpoints, a manually-triggered (then
cron-triggered) retraining script, and a folder-plus-metadata-row model registry will
go a long way. Upgrade individual pieces only when a specific, real pain point (latency,
scale, a compliance audit) forces it — not preemptively.

## 12. Technology stack summary

| Technology | Used for | Why this, specifically |
|---|---|---|
| openEHR / EHRbase | Clinical data storage | Vendor-neutral clinical data model (archetypes/templates) decoupled from application schema; AQL gives a standard query interface over that model |
| PostgreSQL | EHRbase's backing store | EHRbase's supported/required backend |
| Docker Compose | Local EHRbase deployment | Matches the two-service (DB + API) topology with health-checked startup ordering |
| Python 3.11 | Everything in `profling_v2` | Existing project convention |
| `requests` | EHRbase AQL/REST calls | Standard, minimal HTTP client; no need for anything heavier |
| Custom recursive-descent parser | Rule expression evaluation | Avoids `eval()`'s arbitrary-code-execution risk on externally-editable rule strings, while still supporting a real boolean-logic grammar |
| Gymnasium | RL environment interface | De facto standard Python RL API; free compatibility with the broader RL tooling ecosystem; no API key, no network dependency, fully local |
| Stable-Baselines3 | DQN implementation | Most widely used, well-tested Gymnasium-native RL library; the algorithm internals (replay buffer, target network, exploration schedule) aren't the novel part of this system |
| NumPy | Feature normalisation, synthetic data sampling | Standard numerical computing in Python |

## 13. Known gaps & limitations (consolidated)

- No real doctor-correction data — the entire training signal is synthetic
  (`doctor_simulator.py`). This is the single biggest gap between "POC" and
  "production."
- No trained-policy persistence/versioning beyond a single local `.zip` file.
- `patient_history` (cold-start state) is in-memory only — lost on every restart.
- `patient_last_ecg_ischemia` is never populated from EHRbase (hardcoded `False`).
- `patient_last_alcohol` is never populated from EHRbase (hardcoded `"false"`).
- Age/gender/ethnicity come from a hardcoded Python lookup table, not EHRbase, despite
  `MyStera_Onboarding_v1` presumably covering demographics.
- `MyStera_Onboarding_v1` template doesn't appear in the live EHRbase template list,
  despite the file existing in `templates/` — needs verification.
- Conditions/symptoms are keyword-matched against free text, not read from dedicated
  structured fields — brittle by construction.
- No CAD-specific case-study patient exists in the EHRbase dummy data, despite CAD
  being one of the four in-scope diseases.
- No production orchestrator exists yet (§11) — no inference service, no feedback
  capture table, no model registry, no monitoring.

## Appendix A — Repository file map (`profling_v2`)

| File | Role |
|---|---|
| `scoring.py` | EHRbase client, profiling, rules engine, agent-profile adapter, CLI |
| `env/cardiac_env.py` | `MysteraCardiacEnv` — the Gymnasium RL environment |
| `data/sample_profiles.py` | 5 hand-written patients for manual walkthroughs |
| `data/synthetic_patients.py` | Training-scale synthetic patient generator |
| `data/doctor_simulator.py` | Synthetic doctor-correction generator |
| `RL_agent.py` | Manual, human-readable episode walkthrough/simulator (random policy) |
| `train_dqn.py` | DQN training + evaluation script |
| `requirements.txt` | `gymnasium`, `numpy`, `requests`, `stable-baselines3` |
| `models/` | Trained policy checkpoints (gitignored, regenerable) |

## Appendix B — Profile schema reference (agent-facing, 8 fields)

| Field | Type | Normalisation range (if applicable) |
|---|---|---|
| `patient_imc` (BMI) | float | [15, 45] |
| `patient_last_smoking` | bool | — |
| `patient_last_famhistory` | bool | — |
| `patient_last_blood_troponint` | float | [0, 0.1] |
| `patient_last_vitals_systolicbp` | float | [90, 200] |
| `patient_last_blood_ldlcholesterol` | float | [50, 250] |
| `patient_last_blood_hdlcholesterol` | float | [20, 100] |
| `patient_last_ecg_ischemia` | bool | — |

Disease taxonomy: `angina`, `atherosclerosis`, `cardiogenicshock`, `cad`. Label
taxonomy: `no_risk` (0), `unconfirmed_risk` (1), `confirmed_risk` (2).

## Appendix C — Glossary

| Term | Meaning |
|---|---|
| **openEHR** | An open clinical-information-model specification that separates clinical data semantics (archetypes/templates) from application/database schema |
| **Archetype** | A reusable, vendor-neutral definition of a clinical concept (e.g. a blood pressure observation), identified by a stable ID like `openEHR-EHR-OBSERVATION.blood_pressure.v2` |
| **Template** | An application-specific composition of archetypes (e.g. "everything captured at a clinical encounter") |
| **AQL** | Archetype Query Language — openEHR's SQL-like query language for clinical data |
| **LOINC** | A standardised vocabulary/coding system for identifying lab tests and clinical observations |
| **MDP** | Markov Decision Process — the formal framework (states, actions, transitions, rewards) underlying reinforcement learning |
| **Contextual bandit** | A simplified RL setting where each decision is independent (no multi-step consequences) — exactly what `MysteraCardiacEnv`'s single-step episodes are |
| **DQN** | Deep Q-Network — an off-policy RL algorithm that learns the expected value of each action in each state via a neural network and an experience replay buffer |
| **Replay buffer** | Storage of past (state, action, reward, next state) transitions, sampled randomly during training — valid because DQN doesn't require transitions to be used in temporal order |
| **Cold start** | The period before a system has enough history about a specific entity (here, a patient) to act on it responsibly |
| **Gymnasium** | The standard Python interface/library for defining RL environments (`reset`/`step`/action and observation spaces) |

## Appendix D — Running this system

```bash
# EHRbase backend (separate repo)
cd ~/Desktop/mystera-ehrbase && docker compose up -d

# This repo
cd ~/profling_v2
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Score real patients against live EHRbase
.venv/bin/python scoring.py --agent-output profiles.json

# Manual, human-readable episode walkthrough (random policy, no training)
.venv/bin/python RL_agent.py

# Train + evaluate the DQN policy on synthetic data
.venv/bin/python train_dqn.py
```

# Human-Subject Study Design: Evaluating Agency-Preserving Interdependent AI Collaborators

This protocol specifies a rigorous, IRB- and pre-registration-ready user study designed to complement the computational MARL contributions (following the human-AI validation paradigm of Ruhdorfer, Bortoletto, Oei, Penzkofer, & Bulling, ICML '26, and grounded in Xiao et al., CHI '24).

---

## 1. Research Questions & Pre-Registered Hypotheses

* **RQ1 (Task & Fluency Performance):** Does training AI teammates under the **Interdependence (`ADC-MARL`)** paradigm improve human-AI task completion and subjective collaborative fluency compared to standard `Task-Only MARL` and `FCP`?
* **RQ2 (Human Agency vs. Paternalistic Override):** Does Primal-Dual agency/empowerment regularization increase human participants' **Sense of Agency** and eliminate frustrating unilateral AI overrides during *Ability Sharing*?
* **RQ3 (Reducing "Invisible Work" & Cognitive Load):** Does `ADC-MARL` paired with the **JSD-Gated `ProToM` Facilitator** reduce human communicative effort and NASA-TLX workload compared to `Task-Only MARL` and `Always-On` assistive broadcasting?

### Pre-Registered Hypotheses
* **H1 (Collaborative Fluency & Completion):** Participants paired with `ADC-MARL` achieve higher task completion rates and higher Hoffman Collaborative Fluency scores than when paired with `FCP` or `Task-Only MARL` ($p < .01$).
* **H2 (Preserved Sense of Agency):** In *Ability Sharing*, `ADC-MARL` significantly increases Sense of Agency Scale (SoAS) positive-agency ratings and reduces unilateral AI overrides compared to `Task-Only MARL` ($p < .001$).
* **H3 (Reduced Invisible Work):** Participants emit significantly fewer supportive utterances per completed trial with `ADC-MARL` than with `Task-Only MARL`, accompanied by lower NASA-TLX Mental Demand and Frustration ($p < .01$).
* **H4 (Selective Cross-Modal Facilitation):** Under asymmetric sensory masking, `ProToM` achieves non-inferior task completion relative to `Always-On` relay (TOST equivalence margin $\Delta = 5\%$) while significantly lowering NASA-TLX Auditory/Visual Clutter and annoyance ratings ($p < .001$).

---

## 2. Two-Phase Participatory & Experimental Structure

To avoid the ableist pitfalls of disability-simulation-only studies (Bennett & Rosner, CHI '19; Xiao et al., CHI '24, Sec. 5.1), we combine a **controlled within-subjects factorial experiment** with an **authentic ability-diverse participatory evaluation**:

### Phase A: Controlled $2 \times 3$ Mixed-Factorial Experiment ($N = 36$)
* **Design:** Within-subjects across **AI Partner Policy** (`FCP` vs. `Task-Only MARL` vs. `ADC-MARL`) $\times$ **Collaboration Mode** (`Ability Combining` vs. `Ability Sharing`), counterbalanced via a $6 \times 6$ Williams Latin Square to control for learning and fatigue effects.
* **Sample Size Justification (A Priori Power Analysis):** In `G*Power 3.1`, for a repeated-measures ANOVA / linear mixed-effects contrast detecting a medium-to-large effect size ($f = 0.25$, $\alpha = .05$, power $1 - \beta = 0.90$, correlation among repeated measures $r = 0.50$), the required minimum sample is $N = 30$; we recruit **$N = 36$** for complete $6$-order Williams Latin Square balance.
* **Interface Asymmetries (via `study_app/`):**
  * *Condition 1 (Ability Combining — Asymmetric Reach/Perception):* The human participant sees the visual order board (`vis = 1.0`) but controls a reach-constrained avatar (`reach = 0.0` on elevated shelves), collaborating with an AI teammate with `vis = 0.0, reach = 1.0`.
  * *Condition 2 (Ability Sharing — Private Human Goal Ownership):* The human participant holds a private target item preference (`g_B`) under masked shelf labels (`vis = 0.0`, spatial audio/TTS active), collaborating with a sighted AI Provider.
* **Trial Structure:** 1 practice block (3 trials) + 6 experimental blocks $\times$ 8 trials/block = **48 scored trials** (~40 minutes total session duration).

### Phase B: Authentic Mixed-Ability Participatory Validation ($N = 12$: 6 BLV, 6 DHH / Motor-Impaired)
* **Design:** Co-design and ecological evaluation session (60 minutes, compensated above local living-wage accessibility research standards) pairing **Blind/Low-Vision (BLV)** participants (using screen reader / Web Speech TTS in `study_app/`) and **d/Deaf or Hard-of-Hearing (DHH)** or **mobility-impaired** participants with `ADC-MARL` + `ProToM` (`none` vs. `always_on` vs. `protom`).
* **Qualitative Protocol:** Semi-structured interview post-interaction coded via reflexive thematic analysis (Braun & Clarke), probing perceived *access intimacy*, *invisible communicative labor*, and whether the AI respected participant pacing vs. "taking over."

---

## 3. Dependent Variables & Measurement Instruments

### A. Objective Behavioral & Information-Theoretic Telemetry (Logged Automatically by `study_app/`)
1. **Task Completion Rate (%)** & **Time-to-Completion (s / steps)**
2. **Unilateral AI Override Rate (%):** Fraction of *Ability Sharing* trials where the AI checks out without human co-presence/handoff.
3. **Human Supportive ("Invisible Work") Count:** Number of vocal/gesture/button utterances required per trial.
4. **Empirical Human Empowerment $\hat{I}(A_{\text{human}}; S_{t+k} \mid s_t)$ (nats)** & **Shapley Contribution Symmetry**.
5. **`ProToM` Intervention Duty Cycle (%):** Fraction of steps generating an assistive TTS/caption alert.

### B. Validated Post-Block Psychometric Scales (7-Point Likert)
1. **Sense of Agency Scale (Tapal et al., 2017 — Adapted):** Positive Agency ("I was in full control of the final choice") & Negative Agency ("The AI partner acted without my consent").
2. **Raw NASA-TLX (Hart & Staveland, 1988):** Mental Demand, Temporal Demand, Effort, and Frustration subscales.
3. **Human-Robot Collaborative Fluency Scale (Hoffman, 2019):** Perceived team coordination, anticipation, and trust.
4. **Inclusion of Other in the Self (IOS, Aron et al., 1992):** Single-item pictorial measure of perceived interpersonal/human-AI interdependence.
5. **Alert Intrusiveness Subscale (Phase B):** Perceived sensory/screen-reader overload across `Always-On` vs. `ProToM`.

---

## 4. Statistical Analysis Plan

* **Continuous / Behavioral Outcomes:** Analyzed via **Linear Mixed-Effects Models (`LMM`)**:
  $$\text{Outcome}_{ijk} \sim \text{Policy}_i \times \text{Mode}_j + \text{BlockOrder} + (1 + \text{Policy}_i \mid \text{Participant}_k)$$
  with Holm-Bonferroni correction across planned pairwise contrasts (`ADC-MARL` vs. `Task-Only`; `ADC-MARL` vs. `FCP`).
* **Ordinal Likert Ratings:** Analyzed via **Aligned Rank Transform (`ART-ANOVA`, Wobbrock et al., CHI '11)** when normality of residuals is violated (Shapiro-Wilk $p < .05$), reporting partial $\eta_p^2$ and Kendall's $W$.
* **Equivalence Testing for `ProToM`:** Two One-Sided Tests (**TOST**) on task completion rate between `ProToM` and `Always-On` ($\pm 5\%$ equivalence bounds) paired with a superiority test on Alert Load and NASA-TLX Frustration.

---

## 5. Running the Included Interactive Human-AI Study App

This repository ships with a standalone web-based study harness in [`study_app/`](../study_app/) that connects human participants in real time to the trained PyTorch checkpoints and logs trial telemetry + post-block questionnaires to CSV:

```bash
python study_app/server.py --port 8765
# Open http://localhost:8765 in any browser (supports keyboard navigation & Web Speech TTS)
```

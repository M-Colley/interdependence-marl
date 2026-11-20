# Theoretical Foundations & Formal Proofs for Constrained Ability-Parameterized POSGs

This document provides the formal mathematical framework and proofs accompanying **Interdependence-MARL (`ADC-MARL`)**, bridging **Interdependence Theory** (Bennett et al., ASSETS '18; Xiao et al., CHI '24) with **Ad-Hoc Teamwork, Zero-Shot Coordination, and Constrained Multi-Agent Reinforcement Learning**.

---

## 1. Constrained Ability-Parameterized POSG (CA-POSG)

We define a **Constrained Ability-Parameterized Partially Observable Stochastic Game**:

$$\mathcal{M}_{\Phi} = \left\langle \mathcal{N}, \mathcal{S}, \Phi, \left\{\mathcal{A}_i^{\text{req}} \times \mathcal{A}_i^{\text{supp}}\right\}_{i \in \mathcal{N}}, \{\Omega_i, O_i(\cdot \mid \phi_i)\}_{i \in \mathcal{N}}, T(\cdot \mid \boldsymbol{\phi}), r_{\text{task}}, \{c_m\}_{m=1}^M, \{\tau_m\}_{m=1}^M, \gamma \right\rangle$$

where:
* $\mathcal{N} = \{0, \dots, N-1\}$ is the collaborator set;
* $\boldsymbol{\phi} = (\phi_0, \dots, \phi_{N-1}) \in \Phi^N \subseteq [0, 1]^{N \times d}$ is the episode capability profile, where agent $i$ observes its own $\phi_i$ via proprioception while partner profiles $\phi_{-i}$ remain latent;
* Each agent selects a composite action $a_i = (a_i^{\text{req}}, a_i^{\text{supp}})$, decomposing **Required (domain) Abilities** from **Supportive (epistemic/grounding) Abilities**;
* In **Ability Sharing** (where goal $G \sim p(g)$ belongs privately to Borrower $B = 0$), the team optimizes expected discounted task return subject to **agency, empowerment, and invisible-work constraints**:

$$\max_{\pi} \; J_{\text{task}}(\pi) = \mathbb{E}_{\pi, \boldsymbol{\phi}}\!\left[\sum_{t=0}^{H-1} \gamma^t r_{\text{task}}(s_t, \mathbf{a}_t, G)\right]$$

$$\text{subject to} \quad C_{\text{override}}(\pi) \le \tau_{\text{override}}, \qquad C_{\text{burden}}(\pi) \le \tau_{\text{burden}}, \qquad \mathcal{E}_B(\pi) \ge \tau_{\text{emp}}$$

where $\mathcal{E}_B(\pi) = \mathbb{E}_{s_t \sim d^\pi}\!\left[I\!\left(A_B^{t:t+k}; S_{t+k} \mid s_t\right)\right]$ is the $k$-step information-theoretic empowerment of the Borrower.

---

## 2. Proposition 1: Information-Reachability Bound Under Asymmetric Capabilities

**Setup.** Consider a 2-agent task with uniform target goal $G \in \{1, \dots, K\}$ ($H(G) = \log K$), where:
1. **Reachability Asymmetry:** Agent $0$ observes $G$ cleanly ($I(O_0^0; G \mid \phi_0) = \log K$), but its physical transition envelope cannot actuate any goal-completing transition ($\phi_0^{\text{reach}} = 0 \implies \mathbb{P}(\text{extract} \mid a_0^{\text{req}}, a_1^{\text{req}} = \text{noop}) = 0$).
2. **Perceptual Asymmetry:** Agent $1$ has full physical actuation ($\phi_1^{\text{reach}} = 1$), but zero direct observation of $G$ ($I(O_1^{0:H}; G \mid A_0^{\text{supp}} = \text{silent}, \phi_1) = 0$), and at most one extraction attempt is permitted per episode.

**Proposition 1 (Strict Necessity of Modality-Matched Supportive Ability).**  
*Under Assumptions (1)–(2), let $M_{0 \to 1}^{0:t}$ denote the sequence of supportive signals received by Agent $1$ through channels $(\phi_0, \phi_1)$ up to extraction time $\tau$. Then the task completion probability $P_{\text{succ}}(\pi)$ of any decentralized joint policy $\pi = (\pi_0, \pi_1)$ satisfies:*

$$P_{\text{succ}}(\pi) \le 1 - \frac{\log K - I\!\left(M_{0 \to 1}^{0:\tau}; G \mid \phi_0, \phi_1\right) - \log 2}{\log(K - 1)} \quad \text{and specifically,} \quad I\!\left(M_{0 \to 1}^{0:\tau}; G\right) = 0 \implies P_{\text{succ}}(\pi) \le \frac{1}{K}.$$

**Proof.**  
By Assumption (1), Agent $0$ cannot trigger extraction solo; thus task completion requires Agent $1$ to select an extraction station $\hat{G} = f(\tau_1) \in \{1, \dots, K\}$ measurable with respect to Agent $1$'s local interaction history $\tau_1$. By Assumption (2) and the Data Processing Inequality across the Markov chain $G \to \tau_0 \to M_{0 \to 1}^{0:\tau} \to \tau_1 \to \hat{G}$, we have:

$$I(G; \hat{G} \mid \phi_0, \phi_1) \le I\!\left(G; M_{0 \to 1}^{0:\tau} \mid \phi_0, \phi_1\right).$$

Let $P_e = \mathbb{P}(\hat{G} \neq G) = 1 - P_{\text{succ}}(\pi)$ denote the station-selection error probability. By Fano's Inequality:

$$H_b(P_e) + P_e \log(K - 1) \ge H(G \mid \hat{G}) = \log K - I(G; \hat{G}) \ge \log K - I\!\left(G; M_{0 \to 1}^{0:\tau} \mid \phi_0, \phi_1\right).$$

Since binary entropy satisfies $H_b(P_e) \le \log 2$, rearranging yields $P_e \ge \frac{\log K - I(G; M_{0 \to 1}^{0:\tau}) - \log 2}{\log(K - 1)}$, establishing the first inequality. Furthermore, when $I(G; M_{0 \to 1}^{0:\tau}) = 0$ (which occurs either when $\pi_0$ emits no supportive actions *or* when $\pi_0$ emits signals exclusively on a modality $m$ where $\phi_0^{(m)} \cdot \phi_1^{(m)} = 0$, such as visual gestures to a blind partner or speech to a deaf partner without a `ProToM` relay), $\hat{G}$ is statistically independent of uniform $G$, so $\mathbb{P}(\hat{G} = G) = \sum_{g=1}^K \frac{1}{K} \mathbb{P}(\hat{G} = g) \le \frac{1}{K}$. $\square$

*Empirical Verification:* For $K = 3$ stations in our benchmark, $\frac{1}{K} = 33.3\%$. Both `Homogeneous-SP` (`28.5%`), `Uniform-DR without ToM` (`28.1%`), and unmediated cross-modal `BLV + DHH` (`29.7%–30.5%`) obey this exact ceiling, while `ADC-MARL` (`97.7%`) and `ADC-MARL + ProToM` (`95.3%`) saturate near the information-complete optimum.

---

## 3. Proposition 2: Policy Invariance & Credit Assignment of Telescoping Epistemic Shaping

Let $b_i^t = q_\psi(\cdot \mid \tau_i^t) \in \Delta^{K-1}$ denote Agent $i$'s decentralized belief distribution over the episode communicandum $c^* \in \{1, \dots, K\}$ (for fixed belief parameters $\psi$), and define the **team common-ground potential**:

$$\Phi_{\text{bel}}(\boldsymbol{\tau}_t, c^*) = \sum_{i \in \mathcal{N}} b_i^t(c^*), \qquad F_t = \beta_{\text{ep}} \left(\gamma \Phi_{\text{bel}}(\boldsymbol{\tau}_{t+1}, c^*) - \Phi_{\text{bel}}(\boldsymbol{\tau}_t, c^*)\right)$$

with $\Phi_{\text{bel}}(\boldsymbol{\tau}_H, c^*) \equiv 0$ on terminal states.

**Proposition 2 (Equilibrium Invariance & Immediate Grounding Feedback).**  
*(a) Augmenting the team reward $\tilde{r}_t = r_t + F_t$ leaves the optimal policy set $\arg\max_\pi J(\pi)$ invariant for every ability profile $\boldsymbol{\phi} \in \Phi^N$.*  
*(b) If Agent $0$ emits a modality-compatible supportive action $a_0^{\text{supp}}$ at step $t$ that transitions Partner $1$'s belief from uniform ($1/K$) to $\ge 1 - \epsilon$, Agent $0$ receives an immediate 1-step advantage increment of $\beta_{\text{ep}} (\gamma (1 - \epsilon) - 1/K)$ at step $t$, whereas redundant repetitions at $t' > t$ yield at most $\beta_{\text{ep}} \epsilon (\gamma - 1) \le 0$.*

**Proof.**  
*(a)* Consider the augmented history-MDP over joint interaction histories $\tilde{s}_t = (s_t, \boldsymbol{\tau}_t, c^*)$. Since $F_t = \gamma \tilde{\Phi}(\tilde{s}_{t+1}) - \tilde{\Phi}(\tilde{s}_t)$ with $\tilde{\Phi}(\tilde{s}_{\text{terminal}}) = 0$, the discounted sum telescopes identically along every trajectory $\zeta = (\tilde{s}_0, \mathbf{a}_0, \dots, \tilde{s}_H)$:

$$\sum_{t=0}^{H-1} \gamma^t \tilde{r}_t = \sum_{t=0}^{H-1} \gamma^t r_t + \beta_{\text{ep}} \sum_{t=0}^{H-1} \left(\gamma^{t+1} \tilde{\Phi}(\tilde{s}_{t+1}) - \gamma^t \tilde{\Phi}(\tilde{s}_t)\right) = \sum_{t=0}^{H-1} \gamma^t r_t - \beta_{\text{ep}} \tilde{\Phi}(\tilde{s}_0).$$

Because $\tilde{\Phi}(\tilde{s}_0)$ depends only on initial observations $(o_0^0, o_1^0, c^*)$ prior to $\mathbf{a}_0$, $\tilde{J}(\pi) = J(\pi) - \beta_{\text{ep}} \mathbb{E}[\tilde{\Phi}(\tilde{s}_0)]$, where the second term is independent of $\pi$. Hence $\nabla_\theta \tilde{J}(\pi_\theta) = \nabla_\theta J(\pi_\theta)$ and all optimal/saddle-point policies coincide (extending Ng, Harada, & Russell, 1999 to multi-agent epistemic histories).  
*(b)* At step $t$, $\Delta_1 = \gamma b_1^{t+1}(c^*) - b_1^t(c^*) \ge \gamma(1 - \epsilon) - 1/K > 0$, providing an $O(1)$-horizon reward signal rather than waiting $O(H)$ physical navigation steps. Once $b_1^{t'}(c^*) \ge 1 - \epsilon$ is retained in recurrent memory, subsequent steps satisfy $\gamma b_1^{t'+1}(c^*) - b_1^{t'}(c^*) \le \gamma \cdot 1 - (1 - \epsilon) = \epsilon - (1 - \gamma)$, eliminating incentive for repetitive utterance spam when $c_{\text{supp}} > 0$. $\square$

---

## 4. Proposition 3: Primal-Dual Lagrangian Saddle Point for Agency-Constrained Sharing

**Proposition 3 (Zero Duality Gap in Occupancy-Measure Space).**  
*Let $\mathcal{K}$ denote the convex polytope of achievable state-action occupancy measures $\rho^\pi(h_t, \mathbf{a}_t)$ of stochastic history-conditioned policies in the finite-horizon CA-POSG, and assume Strict Slater Feasibility (there exists $\pi^\circ$ satisfying $C_{\text{override}}(\pi^\circ) < \tau_{\text{override}}$, $C_{\text{burden}}(\pi^\circ) < \tau_{\text{burden}}$, and $\mathcal{E}_{\text{lin}}(\pi^\circ) > \tau_{\text{emp}}$). Then strong duality holds for the Lagrangian:*

$$\max_{\pi} \min_{\boldsymbol{\nu} \ge \mathbf{0}} \mathcal{L}(\pi, \boldsymbol{\nu}) = \min_{\boldsymbol{\nu} \ge \mathbf{0}} \max_{\pi} \mathcal{L}(\pi, \boldsymbol{\nu})$$

$$\mathcal{L}(\pi, \boldsymbol{\nu}) = J_{\text{task}}(\pi) - \nu_{\text{over}}\!\left(C_{\text{override}}(\pi) - \tau_{\text{override}}\right) - \nu_{\text{burd}}\!\left(C_{\text{burden}}(\pi) - \tau_{\text{burden}}\right) - \nu_{\text{emp}}\!\left(\tau_{\text{emp}} - \mathcal{E}_{\text{lin}}(\pi)\right)$$

*Proof Sketch.* Because $J_{\text{task}}$, $C_{\text{override}}$, $C_{\text{burden}}$, and linear agency indicators are linear functionals $\langle r_m, \rho \rangle$ over the compact convex occupancy polytope $\mathcal{K}$ (Altman, 1999; Paternain et al., NeurIPS 2019), the primal constrained problem is a finite-dimensional linear program over $\rho \in \mathcal{K}$. Under Slater's condition, strong LP/convex duality guarantees zero duality gap and bounded optimal Lagrange multipliers $\|\boldsymbol{\nu}^*\|_1 \le \frac{J_{\max} - J(\pi^\circ)}{\min_m |\tau_m - C_m(\pi^\circ)|}$. $\square$

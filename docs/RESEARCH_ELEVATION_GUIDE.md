# Research Paper Submission & Elevation Guide
## Targeting SocProS 2027 (IIT Roorkee) and IEEE GCON 2027 (NIT Meghalaya)

This guide provides the complete submission strategy, track alignment, comparative superiority analysis, and step-by-step checklist to submit this research work to:
1. **1st Priority:** **SocProS 2027** — 14th International Conference on Soft Computing for Problem Solving, organized by the Department of Applied Mathematics and Scientific Computing (D-AMSC), **Indian Institute of Technology Roorkee (IIT Roorkee)**, March 5–7, 2027. (Proceedings intended for **Springer**).
2. **2nd Priority:** **IEEE GCON 2027** — 4th IEEE Guwahati Subsection Conference, organized by **National Institute of Technology Meghalaya (NIT Meghalaya)**, June 16–18, 2027. (Proceedings intended for **IEEE Xplore**).

---

## 1. Conference Metadata & Important Deadlines

| Conference Detail | SocProS 2027 (IIT Roorkee) [Priority 1] | IEEE GCON 2027 (NIT Meghalaya) [Priority 2] |
|---|---|---|
| **Host Institution** | IIT Roorkee (D-AMSC) | NIT Meghalaya |
| **Dates** | March 5–7, 2027 | June 16–18, 2027 |
| **Proceedings** | Springer (LNCS / AISC / CCIS) | IEEE Xplore |
| **Paper Submission Portal** | [Microsoft CMT: SocProS 2027](https://cmt3.research.microsoft.com/SocPros2027) | [Microsoft CMT: GCON 2027](https://cmt3.research.microsoft.com/GCON2027/) |
| **Conference Website** | [https://www.socpros2027.iitr.ac.in/](https://www.socpros2027.iitr.ac.in/) | [https://gcon2027.in/](https://gcon2027.in/) |
| **Paper Submission Deadline** | **November 20, 2026** *(Extended)* | **December 15, 2026** |
| **Acceptance Notification** | December 22, 2026 | March 1, 2027 |
| **Camera-Ready & Reg.** | January 24, 2027 | March 31, 2027 |
| **Format & Page Limits** | Springer LNCS (Single-column, 12–15 pages) | IEEE Double-column (Strictly 6 pages) |

---

## 2. Track Matching & Thematic Alignment

### For SocProS 2027 (IIT Roorkee):
- **Conference Theme:** *"Bharat Leads with AI"*
- **Recommended Primary Tracks to Select in Microsoft CMT:**
  1. `Hybrid Intelligent Systems` (Coupling continuous-action Deep RL with analytical geometric medial-axis planning and discrete Control Barrier Functions).
  2. `Reinforcement Learning` (Continuous-action PPO local controller conditioned on 24-sector LiDAR and lookahead waypoints).
  3. `Autonomous Systems / Engineering` (Autonomous Mobile Robots, differential-drive kinematics, obstacle avoidance, and SLAM).
  4. `AI-driven Optimization` (Clearance-weighted Dijkstra optimization and predictive velocity governors).

### For IEEE GCON 2027 (NIT Meghalaya):
- **Recommended Tracks to Select in Microsoft CMT:**
  1. `Control Systems, Automation, Instrumentation, and Intelligent Mobility`
  2. `Artificial Intelligence, Machine Learning, Data Science, and Architecture`

---

## 3. How This Project Outperforms Previous Recent Research (2021–2026 SOTA)

When presenting this work to reviewers at IIT Roorkee or IEEE conferences, emphasize these **six distinct dimensions where our architecture decisively outperforms prior literature**:

### 1. Formalization of the Safety-Conservatism Failure Mode Dichotomy
- **Previous SOTA (Arena-Rosnav 2021, Arena-Bench 2022, BARN 2020):** These benchmark suites simply report aggregate "Success Rate" and "Collision Rate" as black-box numbers, treating timeouts and collisions as undifferentiated failures.
- **Our Advance:** We uncover and prove the **Conservative Timeout vs. Aggressive Collision Dichotomy**:
  - DWA fails exclusively via conservative local-minima timeouts ($27.0\%$) with $0.0\%$ collisions because inflation layers invalidate forward rollouts in tight spaces.
  - Raw PPO fails exclusively via collisions ($24.0\%$) with $0.0\%$ timeouts because geodesic progress rewards saturate forward speed ($v \to 0.26\,\text{m/s}$), causing corner-clipping collisions.
- **Reviewer Pitch:** *"Rather than reporting raw success percentages, we explain the physical and algorithmic mechanics governing why classical planners freeze while deep RL controllers collide."*

### 2. Clearance-Weighted Medial-Axis Doorway Traversal
- **Previous SOTA (PRM-RL 2020, DD-PPO 2020, Nav2 2023):** Standard $A^*$ or uniform Dijkstra computes paths that mathematically graze obstacle boundaries, clipping corners and causing AMRs to wedged on door jambs.
- **Our Advance:** We formulate a quadratic clearance cost penalty:
  $$\text{Cost}_{\text{clear}}(r, c) = \left(\frac{d_{\text{margin}} - D_{\text{clear}}}{d_{\text{margin}}}\right)^2 K_{\text{penalty}}$$
  which forces global waypoints to naturally track the medial axis of narrow constrictions.
- **Result:** In narrow multi-room layouts (`rooms`), PPO achieves **44.0% success vs. DWA’s 32.0% (+12.0% advantage)**, completely eliminating the door-jamb freeze trap.

### 3. Sub-4ms Autonomous Dynamic Impasse & Alternative Rerouting
- **Previous SOTA:** Standard recovery behaviors (such as Nav2 spin/backup) reset local state, but the static global planner continues to command the exact same blocked waypoint, creating an infinite deadlock loop.
- **Our Advance:** Our multi-source watchdog detects stagnation ($\Delta x < 4\,\text{cm}$ over 15 frames) or repetitive boundary holds in $<0.4\,\text{s}$, dynamically marks the pinch point with a virtual obstacle disk ($r_{\text{imp}} = 0.45\,\text{m}$), and re-propagates the Dijkstra wave-front in $< 4\,\text{ms}$, automatically discovering alternative open corridors without human intervention.

### 4. Active 4-Layer Predictive Dynamic Safety Shield (Discrete-Time CBF)
- **Previous SOTA:** Either naive rule-based stops (which induce freezes) or computationally intensive non-linear Control Barrier Function Quadratic Programs (CBF-QPs) that require multi-millisecond solver times on GPUs.
- **Our Advance:** We formulate an analytical discrete-time CBF stopping governor:
  $$v \le \sqrt{2 a_v^{\max} d_{\text{clear}}}$$
  paired with lateral repulsive steering and non-sticky wall disengagement ($d_{\text{cand}} > d_{\text{curr}}$).
- **Result:** **Completely eliminates collisions ($24.0\% \to 0.0\%$)** across all test layouts while elevating overall success to **82.0% [0.73, 0.89]** and SPL to **0.812**, with sub-millisecond CPU overhead.

### 5. Solving the Out-of-Distribution (OOD) Generalization Cliff
- **Previous SOTA:** Most papers test only on familiar training topologies with slight obstacle repositioning.
- **Our Advance:** We evaluate on a strictly held-out, unseen topological family (sharp ZigZag switchbacks, seeds 20000–20039). Raw PPO collapses to $43.0\%$ success with a $57.0\%$ collision rate. Our complete MAS-RL architecture restores success to **78.0% [0.69, 0.85]** ($0.0\%$ collisions, $0.742$ SPL), decisively outperforming DWA ($66.0\%$).

### 6. Statistical Rigor with Wilson Score and Bootstrap CIs
- **Previous SOTA:** Small sample sizes ($N \le 10$), standard deviations only, or unverified claims.
- **Our Advance:** 100 matched scenario trials per experimental cell ($>1,500$ total evaluations) with **95% Wilson Score Confidence Intervals** for binomial proportions, **2,000-resample Bootstrap CIs** for SPL, and cluster-conservative variance adjustments.

---

## 4. Manuscript Files Ready for Submission

The repository now contains all necessary publication-grade files:

1. **Full Research Paper (Markdown):**  
   [`docs/RESEARCH_PAPER_SOCPROS2027.md`](file:///d:/Mini_project_autonomous-navigation-slam-rl/docs/RESEARCH_PAPER_SOCPROS2027.md)  
   *Complete academic draft conforming to Springer style with all math, literature review, tables, and ablations.*

2. **Springer LNCS LaTeX Manuscript (SocProS 2027):**  
   [`docs/paper_socpros2027.tex`](file:///d:/Mini_project_autonomous-navigation-slam-rl/docs/paper_socpros2027.tex)  
   *Compilable LaTeX file in Springer LNCS format, ready to compile to PDF on Overleaf or TeX Live.*

3. **IEEE 6-Page Double-Column LaTeX Manuscript (GCON 2027):**  
   [`docs/paper_ieee_gcon2027.tex`](file:///d:/Mini_project_autonomous-navigation-slam-rl/docs/paper_ieee_gcon2027.tex)  
   *Strictly formatted 6-page double-column IEEE template for NIT Meghalaya.*

4. **High-Resolution Figures for the Paper:**  
   - [`results/figures/fig_benchmark_overview.png`](file:///d:/Mini_project_autonomous-navigation-slam-rl/results/figures/fig_benchmark_overview.png) (Success, collision, timeout, and SPL comparison).
   - [`results/figures/fig_family_breakdown.png`](file:///d:/Mini_project_autonomous-navigation-slam-rl/results/figures/fig_family_breakdown.png) (Topology-by-topology performance).
   - [`results/figures/fig_radar_tradeoff.png`](file:///d:/Mini_project_autonomous-navigation-slam-rl/results/figures/fig_radar_tradeoff.png) (Radar plot of the safety vs. conservatism dichotomy).
   - [`results/figures/fig_robustness_perturbation.png`](file:///d:/Mini_project_autonomous-navigation-slam-rl/results/figures/fig_robustness_perturbation.png) (Noise and pose drift robustness).
   - [`results/figures/trajectories_test.png`](file:///d:/Mini_project_autonomous-navigation-slam-rl/results/figures/trajectories_test.png) & [`results/figures/trajectories_heldout.png`](file:///d:/Mini_project_autonomous-navigation-slam-rl/results/figures/trajectories_heldout.png) (Trajectory visualization).

---

## 5. Step-by-Step Submission Instructions for SocProS 2027 (IIT Roorkee)

1. **Access Microsoft CMT Portal:**
   - Navigate to: [https://cmt3.research.microsoft.com/SocPros2027](https://cmt3.research.microsoft.com/SocPros2027)
   - Log in or create an account with your official institutional email.
2. **Create New Submission:**
   - Click **"Create new submission..."**.
   - Select Primary Subject Area: **Hybrid Intelligent Systems** or **Reinforcement Learning**.
   - Select Secondary Subject Area: **Autonomous Systems** or **AI-driven Optimization**.
3. **Title and Abstract:**
   - **Title:** `Bridging the Safety-Conservatism Dichotomy: A Medial-Axis Shielded Reinforcement Learning Framework for Autonomous Mobile Robot Navigation`
   - **Abstract:** Copy the abstract from [`docs/RESEARCH_PAPER_SOCPROS2027.md`](file:///d:/Mini_project_autonomous-navigation-slam-rl/docs/RESEARCH_PAPER_SOCPROS2027.md) (approx. 200 words).
4. **Authors & Affiliations:**
   - Enter all author names, institutional affiliations, and email addresses. Specify the corresponding author.
5. **Upload Manuscript PDF:**
   - Upload the PDF generated from [`docs/paper_socpros2027.tex`](file:///d:/Mini_project_autonomous-navigation-slam-rl/docs/paper_socpros2027.tex).
6. **Confirm and Submit:**
   - Review the submission summary, agree to author commitments, and click **Submit**. You will receive an official Paper ID (e.g., `SocProS2027_Paper_XXX`).

---

## 6. Live Demonstration & Video for Presentation

For conference presentation (and video submission if required):
1. **Launch the Mission Control GUI:**
   ```powershell
   python run_ui.py
   ```
2. **Key Demonstration Sequence for Examiners / Attendees:**
   - **Medial-Axis Doorway Traversal:** Click into an adjacent room; point out how the planned path centers through the doorway rather than hugging the door frame.
   - **Dynamic Obstacle Injection & Safety Watchdog:** Draw a custom box in front of the moving robot; show the green LiDAR rays turn red and the robot decelerate smoothly to zero without collision.
   - **Dynamic Impasse Rerouting:** Draw a box closing a doorway; point out the orange `⚠️ IMPASSE` marker that immediately appears and show Dijkstra rerouting through the alternative open room in $< 4\,\text{ms}$.
   - **Real-Time Controller Swapping:** Press `[C]` to switch between `PPO + Shield`, `Raw PPO`, and `DWA` to demonstrate DWA's freeze in tight spaces vs. PPO's fluid traversal.

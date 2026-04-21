**A Research Project on Spatially-Grounded World Models**
Version 0.3 · April 2026 · Maxime Zhu

---

## 1. What This Is

Atmosphere.ai is a research project on **anchored world model generation**: using real-world spatial data (GIS geometry, street-level imagery) to ground the outputs of open-source generative world models, so that generated scenes correspond to real locations rather than plausible-looking inventions.

This is not a product. The goal is a body of research — papers, open-source artifacts, public benchmarks — that establishes presence in the intersection of AEC and generative world models. Commercial viability is explicitly out of scope.

For technical architecture, see `atmosphere_spec.md`.

---

## 2. Thesis

> Open-source generative world models can be meaningfully grounded in real-world spatial data through a retrieval-and-conditioning protocol, such that the resulting scenes are verifiable against the physical locations they claim to represent.

Three testable claims:

- **C1.** Current open-source world models (Lyra 2.0, Hunyuan World 2.0, WorldGen) produce scenes that are internally consistent but externally ungrounded — they look like places, but correspond to no specific place.
- **C2.** It is possible to inject geographic spatial priors (GIS building footprints, DEM terrain, street-level imagery) into the generation pipeline in a way that measurably improves correspondence to a real location.
- **C3.** The fidelity gain can be quantified by a novel evaluation protocol that compares generated scenes against held-out real-world observations of the same coordinate.

---

## 3. Why Now

Three things converged in early 2026 that made this project feasible:

- **Open-source world models became good enough.** Hunyuan World 2.0 (April 2026) claims parity with closed-source Marble. Lyra 2.0 (April 2026) is Apache 2.0 and exposes a geometry-based retrieval mechanism internally. WorldGen runs end-to-end on a consumer GPU.
- **Lyra 2.0 introduced external geometric retrieval as a first-class concept** — for its own history frames. Generalizing this to external GIS data is a natural next step that no one has published yet.
- **The evaluation problem is unsolved.** "How grounded is a generated scene?" has no standard metric. Defining one is an independent research contribution.

---

## 4. Technical Positioning

Two backbones serve different purposes:

| Role | WorldGen | Lyra 2.0 |
|------|----------|----------|
| Purpose | Prove the end-to-end pipeline runs on modest hardware | Investigate external geometric retrieval in a video-diffusion world model |
| Modification | Pre-generation conditioning via SCP bundle | Replace internal geometry retriever with external GIS retriever |
| Paper this feeds | System paper (ACADIA / SimAUD) | Core research paper (CVPR/SIGGRAPH workshop or 3DV) |

Marble is used as a closed-source upper-bound reference only. It is not modified.

Contributions, ordered by ambition:

1. **Spatial Conditioning Protocol (SCP)** — a versioned bundle format defining how geographic priors are encoded before reaching a world model. Model-agnostic.
2. **Grounded Generation Pipeline (GGP)** — concrete implementation across both backbones.
3. **Spatial Fidelity Benchmark (SFB)** — held-out evaluation dataset + four metrics + public leaderboard.

---

## 5. Computer Vision Research Spine

Four CV problems, in decreasing certainty:

- **RQ-Eval** — How do you define "grounded" as a measurable CV metric? (This is SFB; it's the infrastructure the other RQs depend on.)
- **RQ-Retrieval** — Given a 3D geometric query, how do you retrieve the most relevant images from a sparse public visual database? Cross-modal contrastive learning between rendered depth and real RGB. (CV Research Point #1.)
- **RQ-Inject** — Can you replace Lyra 2.0's internal geometry retriever with an external GIS retriever while preserving long-horizon consistency? (CV Research Point #4, highest risk/reward.)
- **RQ-Localize** — Can you estimate camera pose in a real city from a single photo + LoD2 priors? Sparse-geometry visual localization. (CV Research Point #2, application-oriented side branch.)

The CV course at Northeastern (Pattern Recognition + Computer Vision) is an opportunity to develop **RQ-Eval** as a course project, and **RQ-Retrieval** or **RQ-Inject** as independent study / thesis directions.

---

## 6. Output Plan

Twelve to eighteen months. Each artifact is independently valuable.

| # | Artifact | Venue |
|---|----------|-------|
| A1 | System paper on SCP + WorldGen pipeline | ACADIA 2026 / SimAUD |
| A2 | Core paper on external geometric retrieval (Lyra variant) | CVPR/SIGGRAPH workshop, or 3DV |
| A3 | Open-source SCP spec + reference implementation | GitHub |
| A4 | Spatial Fidelity Benchmark + leaderboard | Hugging Face dataset |
| A5 | Technical blog series | Personal site + LinkedIn |
| A6 | Reviewer role | ACADIA / SimAUD, 2027 cycle |

---

## 7. Phased Plan

- **Phase 0 · PoC · 4 weeks.** End-to-end: coordinate → SCP bundle → WorldGen output → side-by-side vs unconditioned baseline. Answers RQ-Eval feasibility and a first read on retrieval value.
- **Phase 1 · Benchmark · 6 weeks.** SFB v1: 20 Seattle coordinates, 4 metrics, baseline scores. Artifact A4 drafted.
- **Phase 2 · Lyra modification · 10 weeks.** Replace internal retriever with external GIS retriever. Core of A2.
- **Phase 3 · Generalization · 12 weeks.** Second city (Nanjing). Polish A1–A4 for submission.

Each phase has a go/no-go gate. A null result at any gate is publishable at workshop level and is not a project failure.

---

## 8. Alignment with Target Roles

This project is designed to produce work that is legible to two audiences simultaneously:

- **AEC / architecture-computing venues** (ACADIA, SimAUD, CAADRIA) — the domain resonance
- **CV / generative modeling teams at autonomous-driving-scale labs** (Waymo Research, NVIDIA Spatial Intelligence, Wayve) — the technical resonance

The technical decisions in `atmosphere_spec.md` — diffusion/flow matching backbones, cross-modal contrastive retrieval, external conditioning of video world models — overlap ~70–80% with the requirements of Waymo's 4D world simulation research roles. This overlap is deliberate.

---

## 9. What's Not in This Document

- Detailed architecture and module interfaces → `atmosphere_spec.md`
- Week-by-week Phase 0 plan → `atmosphere_spec.md` §6
- Risk analysis → deferred to post-Phase 0 retrospective (risks before you start are speculation)
- Commercial extensions → out of scope
- Open questions → they belong in the code's `TODO` comments and the project's GitHub issues, not here

---

*v0.3 replaces v0.2 (April 2026). Expected next revision after Phase 0 completion, when real results will replace speculation.*

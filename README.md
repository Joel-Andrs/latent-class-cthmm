# Latent-class continuous-time hidden Markov models for retinal fluid dynamics: trial-to-routine-care prior transfer

**Status: exploratory prototypes (September 2026).** These are the Stan programs behind the sequential prior-transfer pipeline described under *Borrowing (II)* in the accompanying research plan. They implement the baseline that Article II of the planned dissertation replaces: a two-phase fit in which a trial source is modelled first and its posterior is reduced to independent priors for a routine-care fit. They are published so that the model described in the plan can be read as code. They are not the multisource-exchangeability extension, which does not yet exist.

No data are included. The trial source is the CATT data set (Comparison of AMD Treatments Trials), obtainable from its public repository; the routine-care source is the HUS doctoral study cohort — routine-care records collected for a clinician's doctoral study at Helsinki University Hospital — which cannot be redistributed. The label `Finnish` / `finland` in file and variable names is the identifier used for the HUS cohort in the code.

## Files

| File | Role |
|---|---|
| `catt_train_hmm_fast.stan` | Phase 1. Latent-class CT-HMM on the trial source. |
| `finland_deploy_hmm_fast_nogap.stan` | Phase 2. The same model on the routine-care source, with transferred priors and an absorbing drop-out state. Treat-and-extend visit-scheduling component ablated (`nogap`). |

This repository contains the two model programs only. The R code that prepares the data, fits Phase 1, aligns class labels, builds the Phase 2 priors and fits Phase 2 is not included here; it is described in *The two-phase transfer* below and will be released with the articles. The data each program expects is specified in full under *Data expected by the Stan programs*, so the models can be read, compiled and driven from any interface without it.

## The model

### State space

Two binary fluid markers, intraretinal fluid (IRF) and subretinal fluid (SRF), define four latent states:

| State | IRF | SRF |
|---|---|---|
| 1 Dry | absent | absent |
| 2 IRF | present | absent |
| 3 Both | present | present |
| 4 SRF | absent | present |

The latent state evolves as a continuous-time Markov jump process. Transitions are allowed only between states that differ in one marker, which puts them on the ring Dry – IRF – Both – SRF – Dry. The two diagonals (Dry ↔ Both, IRF ↔ SRF), in which both markers would change at the same instant, are fixed at zero. This is a substantive assumption — compartments appear and resolve one at a time — and a testable one: freeing the four diagonal entries gives a nested twelve-intensity model that can be compared on predictive score using the `log_lik` output.

### Generator

The eight intensities are class-specific and log-linear in age group:

| `beta_fluid[c, k]` | Transition | `Q` entry |
|---|---|---|
| k = 1 | Dry → IRF | `Q[1,2]` |
| k = 2 | IRF → Dry | `Q[2,1]` |
| k = 3 | IRF → Both | `Q[2,3]` |
| k = 4 | Both → IRF | `Q[3,2]` |
| k = 5 | Both → SRF | `Q[3,4]` |
| k = 6 | SRF → Both | `Q[4,3]` |
| k = 7 | SRF → Dry | `Q[4,1]` |
| k = 8 | Dry → SRF | `Q[1,4]` |

`Q[c, a][from, to] = exp(beta_fluid[c, k] + gamma_age_fluid[a − 1][k])` for age group `a > 1`, and `exp(beta_fluid[c, k])` for the reference group. Diagonals are set so that rows sum to zero. Intensities are per week.

### Latent classes

`N_classes` classes (three in the exploratory runs). Each class has its own eight log-intensities and its own initial-state distribution `pi_init[c]`; `theta_class` is the class prevalence. The age effects are shared across classes. Classes are exchangeable in the likelihood, so labels are identified only up to permutation within a fit; alignment is done after sampling (see below).

### Observation model

Each marker reading is a noisy binary observation of the corresponding component of the latent state. For each marker an ordered pair of logits gives the false-positive and true-positive rates: `theta_srf_logit[1]` is the logit of P(read present | truly absent), `theta_srf_logit[2]` the logit of P(read present | truly present), and likewise for IRF. The constraint FP < TP, together with priors anchored near 12% and 88%, separates reading error from genuine transitions — the classical weak spot of hidden Markov models. A reading coded `-1` means the marker was not recorded at that visit and contributes nothing to the likelihood; the visit still contributes its time gap.

### Drop-out (Phase 2 only)

Routine care ends follow-up for reasons that may depend on the disease. The routine-care model adds an absorbing fifth state with a class-specific log-intensity from each fluid state, `beta_dropout[c, 1..4]`, and its own age effects. The pipeline sets `is_dropout = 1` on a patient's terminal visit when follow-up ended for a recorded reason; at that visit the emission puts all mass on state 5. A patient without the indicator is right-censored at the last visit, and because state 5 is excluded at every observed visit, the probability of not having dropped out is part of the likelihood: drop-out is informative, not ignored. The trial program has no drop-out state because the exploratory trial cohort — the as-needed arm, chosen as the regimen closest to routine care — was defined without informative drop-out. A production fit on the full trial would add one and treat trial attrition as Phase 2 treats drop-out.

### Likelihood and computation

The likelihood is the forward algorithm in log space: for patient *p* and class *c*, the forward vector is initialised at `log pi_init[c]`, propagated across each observed gap by the transition matrix `P(dt) = expm(Q[c, a] · dt)`, and updated by the emission of each visit. The class mixture is taken outside the recursion, `p(y_p) = Σ_c theta_c p(y_p | c)`, so `log_mix_pat[p, c]` holds the log of each term and `log_lik[p]` their log-sum. Posterior class membership is `class_post[p] = softmax(log_mix_pat[p])`.

Two caches keep the cost manageable: emission log-probabilities are computed once per visit, and `P(dt)` is computed once per (class, age group, unique gap) rather than once per visit. With three classes, three age groups and 89 distinct gaps in the routine-care data, that is 801 matrix exponentials of a 5 × 5 generator per log-density evaluation. Sampling is by Hamiltonian Monte Carlo in Stan.

## The two-phase transfer

The procedure the two programs are designed for. Steps 1 and 4 are the programs in this repository; steps 2 and 3 are R code held outside it.

1. **Fit Phase 1** on the trial source. The exploratory runs used the as-needed (PRN) arm — the regimen closest to routine care, under a cohort definition with no informative drop-out — two years of monthly visits, three classes, and asymmetric initial values (slow / moderate / fast classes) to encourage separation.
2. **Align class labels.** Chains and iterations may permute the classes. The pipeline picks the maximum-a-posteriori draw as pivot and, for each draw, chooses the permutation of classes that minimises the squared distance of `beta_fluid` to the pivot (pivotal reordering). The same permutation is applied to `pi_init` and `theta_class`. Convergence diagnostics (R-hat, bulk ESS) are computed on the aligned draws. The age effects need no alignment because they are shared across classes.
3. **Reduce to priors.** Elementwise means and standard deviations of the aligned `beta_fluid` and of `gamma_age_fluid` become independent normal priors. The mean of each class's `pi_init` is scaled by a fixed pseudo-count of 10 to give Dirichlet parameters.
4. **Fit Phase 2** on the routine-care source with those priors. Class prevalence, reading error and the drop-out process are estimated afresh, without transfer.

What this transfer does and does not do matters for the research plan. It carries the trial's information into the routine-care fit, but the strength of borrowing is set by construction: the normal priors on the log-intensities and age effects are fixed at the trial's posterior width, and the initial-state Dirichlet at a fixed pseudo-count of 10, so no prior can respond to conflict between the sources; it discards all posterior correlation between intensities; and it carries class identity across sources only through the location of the priors, with nothing in the parameterisation enforcing that class *c* means the same thing in both fits. It cannot test whether pooling is warranted. That is the limitation Article II addresses by putting a mixture prior on each intensity and estimating the pooling weight.

## Relation to section 4 of the research plan

| Plan, §4 | Here |
|---|---|
| "continuous-time Markov processes on a latent state space, observed irregularly and with error" | Generator `Q`, arbitrary gaps `unique_dt`, emission model with FP/TP logits |
| "matrix exponentials of the generator over observed gaps … Hamiltonian Monte Carlo in Stan" | `matrix_exp(Q * unique_dt[u])`, cached per (class, age group, gap) |
| "Four states … ring … Dry–Both and IRF–SRF fixed at zero, leaving eight intensities" | `Q` construction; entries `[1,3]`, `[3,1]`, `[2,4]`, `[4,2]` never assigned |
| "nested twelve-intensity model compared by predictive score" | Free the four diagonal entries; compare on `log_lik` |
| "class-specific over three latent classes, age enters as group effects, marker error by ordered false- and true-positive logits" | `beta_fluid[N_classes, 8]`, `gamma_age_fluid`, `ordered[2] theta_*_logit` |
| "sequential transfer: the trial is fitted first, label switching resolved by permutation alignment, aligned posterior reduced to independent normal priors for the routine-care fit, which alone models drop-out" | Steps 1–4 above; `prior_*` data in Phase 2; state 5 |
| "Pooling strength is fixed and untestable" | Prior standard deviations are data, not parameters |
| Article II: per-intensity mixture prior, marginalised indicator, estimated weight; ordering constraint fixing class labels | Not implemented; planned |
| Article III: random effects on log-intensities, out-of-source calibration | Not implemented; planned |
| Article IV: treatment rules, forward simulation | Not implemented; the prototypes contain no treatment covariate |

## Known limitations

- **Fixed borrowing.** As above. The Phase 2 priors are also computed from a 400-draw scouting run (100 post-warm-up draws per chain); their standard deviations are noisy, and a production-length Phase 1 fit is scheduled before any transferred prior is used for reported results.
- **No treatment covariate.** Injections are recorded in the pipeline but do not enter the model. The fitted process is marginal over the injection schedule of each source; a model of treatment switching (Article IV) needs the intensities to depend on treatment.
- **Drop-out dating.** The drop-out visit is the patient's real terminal visit, so its fluid readings are discarded and the transition into state 5 is dated within the last observed interval. A cleaner convention keeps the terminal readings and appends a pseudo-visit for absorption at a chosen interval after them.
- **Visit clustering.** The pipeline merges routine-care visits less than a week apart, comparing each visit with its immediate predecessor, so a chain of close visits can be merged into one cluster spanning more than a week. Comparing with the first visit of the current cluster bounds the span.
- **Age groups.** Three groups (≤ 75, 76–85, ≥ 86) with the first as reference; missing age is currently assigned to the middle group.
- **Visit scheduling.** Routine-care visit times are treated as given (`nogap`), which is valid when the next gap depends only on readings already observed and not on the latent state. The full routine-care model includes a treat-and-extend scheduling component in which the gap to the next visit depends on the fluid state; it is not included here.
- **Class identity across sources** is carried only by the transferred priors.

## Data expected by the Stan programs

| Field | Type | Meaning |
|---|---|---|
| `N_obs`, `N_pat` | int | visits, patients |
| `start_idx`, `end_idx` | int[N_pat] | first and last row of each patient; rows are patient-contiguous and time-ordered |
| `N_classes` | int | latent classes |
| `N_unique_dt`, `unique_dt` | int, vector | distinct positive gaps (weeks) |
| `dt_idx` | int[N_obs] | index into `unique_dt`; 0 on each patient's first visit |
| `irf`, `srf` | int[N_obs] | 1 present, 0 absent, −1 not recorded |
| `N_age_groups`, `pat_age_group` | int, int[N_pat] | age group per patient, 1 = reference |
| `is_dropout` | int[N_obs] | Phase 2 only; 1 on a terminal visit ending follow-up |
| `prior_beta_fluid_mu`, `prior_beta_fluid_sd` | matrix[N_classes, 8] | Phase 2 only; aligned Phase 1 summaries |
| `prior_gamma_fluid_mu`, `prior_gamma_fluid_sd` | matrix[N_age_groups − 1, 8] | Phase 2 only |
| `prior_pi_init_alpha` | vector[4][N_classes] | Phase 2 only; Dirichlet parameters |

The `transformed data` blocks check the layout (first visit has `dt_idx == 0`; `is_dropout` only on a terminal visit, never on the first) and stop with a message if it is violated.

## Running

Compile each program once with `rstan::stan_model` or `cmdstanr::cmdstan_model`; both use the current array syntax and need Stan ≥ 2.33 or rstan ≥ 2.32. Supply the fields listed above; Phase 2 additionally needs the four `prior_*` fields produced by steps 2 and 3.

The exploratory runs used four chains and asymmetric initial values for `beta_fluid` (one slow, one moderate, one fast class) to encourage class separation, since a symmetric start leaves the classes unidentified in the early iterations. Exclude `log_mix_pat` and `class_post` from saved draws with `pars` if output size matters; `log_lik` is what `loo` needs.

## Changes from the versions used in the exploratory runs

Relative to the files used in the exploratory runs, the published programs: expose `log_lik`, `log_mix_pat` and `class_post` by moving the likelihood into `transformed parameters` (the computation itself is unchanged); generalise the age-effect arrays from a hard-coded two to `N_age_groups − 1`; add `transformed data` checks of the data layout; tighten data bounds on gaps, prior standard deviations and Dirichlet parameters; and replace working comments with documentation. Generator entries, emission mapping, priors and the forward recursion are identical to the exploratory versions.

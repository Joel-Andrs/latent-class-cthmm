// =============================================================================
// catt_train_hmm_fast.stan
// Phase 1 of the two-phase prior-transfer prototype: latent-class
// continuous-time hidden Markov model (CT-HMM) fitted to the trial source.
//
// Latent process
//   A continuous-time Markov jump process on four fluid states,
//     1 = Dry (no IRF, no SRF)   2 = IRF only   3 = Both   4 = SRF only,
//   with transitions on the ring Dry - IRF - Both - SRF - Dry. The two
//   diagonals Dry <-> Both and IRF <-> SRF, in which both markers would change
//   at once, are structurally zero, leaving eight intensities.
//
// Heterogeneity
//   N_classes latent classes. Each class has its own vector of eight
//   log-intensities (beta_fluid[c]) and its own initial-state distribution
//   (pi_init[c]); theta_class is the class prevalence. Age enters as additive
//   group effects on the log-intensities, shared across classes; group 1 is
//   the reference and carries no parameter.
//
// Observation model
//   Each marker is read with error. For marker m, the ordered pair
//   theta_m_logit = (false-positive logit, true-positive logit) gives
//     P(read present | truly absent) and P(read present | truly present).
//   The constraint FP < TP is what identifies the error model against the
//   transition model. A reading coded -1 (not recorded) contributes nothing.
//
// Likelihood
//   Forward algorithm in log space. Transition matrices P(dt) = expm(Q * dt)
//   are computed once per (class, age group, unique gap) and cached, so the
//   cost is N_classes * N_age_groups * N_unique_dt matrix exponentials per
//   log-density evaluation rather than one per visit. Drop-out is not
//   modelled in the trial source (protocol follow-up); the routine-care model
//   in finland_deploy_hmm_fast_nogap.stan adds it.
//
// Outputs
//   log_lik[p]        marginal log-likelihood of patient p (for predictive
//                     model comparison, e.g. against the nested
//                     twelve-intensity model with the diagonals freed).
//   log_mix_pat[p, ]  unnormalised log posterior class weights of patient p.
//   class_post[p]     posterior class membership of patient p (softmax).
//
// Status: exploratory prototype (September 2026). This is the sequential
// prior-transfer baseline, in which pooling strength between sources is fixed
// by construction; it is not the multisource-exchangeability extension
// described in the research plan.
// =============================================================================

data {
  // ---- observations, stored patient-contiguously ---------------------------
  int<lower=1> N_obs;                          // total visits
  int<lower=1> N_pat;                          // patients
  array[N_pat] int<lower=1, upper=N_obs> start_idx;   // first row of patient p
  array[N_pat] int<lower=1, upper=N_obs> end_idx;     // last row of patient p
  int<lower=1> N_classes;                      // latent classes

  // ---- time gaps, indexed for matrix-exponential caching -------------------
  // dt_idx[i] = 0 for the first visit of a patient (no transition);
  // otherwise unique_dt[dt_idx[i]] is the gap (weeks) since the previous visit.
  int<lower=1> N_unique_dt;
  vector<lower=0>[N_unique_dt] unique_dt;
  array[N_obs] int<lower=0, upper=N_unique_dt> dt_idx;

  // ---- marker readings: 1 = present, 0 = absent, -1 = not recorded ---------
  array[N_obs] int<lower=-1, upper=1> irf;
  array[N_obs] int<lower=-1, upper=1> srf;

  // ---- patient-level covariate: age group (1 = reference) -----------------
  int<lower=1> N_age_groups;
  array[N_pat] int<lower=1, upper=N_age_groups> pat_age_group;
}

transformed data {
  // Structural checks on the data layout the likelihood relies on.
  for (p in 1:N_pat) {
    if (start_idx[p] > end_idx[p])
      reject("patient ", p, ": start_idx exceeds end_idx");
    if (dt_idx[start_idx[p]] != 0)
      reject("patient ", p, ": first visit must have dt_idx == 0");
  }
  for (u in 1:N_unique_dt) {
    if (unique_dt[u] <= 0)
      reject("unique_dt[", u, "] must be positive");
  }
}

parameters {
  simplex[N_classes] theta_class;              // class prevalence

  // Observation error, one ordered pair per marker: [1] false-positive logit,
  // [2] true-positive logit. 'ordered' enforces FP < TP.
  ordered[2] theta_srf_logit;
  ordered[2] theta_irf_logit;

  // Class-specific baseline log-intensities for the eight ring transitions,
  // in the order used in the Q construction below.
  matrix[N_classes, 8] beta_fluid;

  // Age-group effects on the log-intensities (group 1 is the reference).
  array[N_age_groups - 1] vector[8] gamma_age_fluid;

  // Class-specific initial-state distributions.
  array[N_classes] simplex[4] pi_init;
}

transformed parameters {
  vector[N_pat] log_lik;                       // per-patient marginal log-lik
  matrix[N_pat, N_classes] log_mix_pat;        // per-patient log class weights
  {
    // ---- 1. Emission log-probabilities, once per visit -------------------
    // Column = latent state. SRF is present in states 3, 4; IRF in states 2, 3.
    matrix[N_obs, 4] log_em_cache = rep_matrix(0.0, N_obs, 4);
    for (i in 1:N_obs) {
      if (srf[i] != -1) {
        real lp_srf_absent  = bernoulli_logit_lpmf(srf[i] | theta_srf_logit[1]);
        real lp_srf_present = bernoulli_logit_lpmf(srf[i] | theta_srf_logit[2]);
        log_em_cache[i, 1] += lp_srf_absent;
        log_em_cache[i, 2] += lp_srf_absent;
        log_em_cache[i, 3] += lp_srf_present;
        log_em_cache[i, 4] += lp_srf_present;
      }
      if (irf[i] != -1) {
        real lp_irf_absent  = bernoulli_logit_lpmf(irf[i] | theta_irf_logit[1]);
        real lp_irf_present = bernoulli_logit_lpmf(irf[i] | theta_irf_logit[2]);
        log_em_cache[i, 1] += lp_irf_absent;
        log_em_cache[i, 2] += lp_irf_present;
        log_em_cache[i, 3] += lp_irf_present;
        log_em_cache[i, 4] += lp_irf_absent;
      }
    }

    // ---- 2. Generator matrices per (class, age group) ----------------------
    // Ring: 1<->2 (Dry, IRF), 2<->3 (IRF, Both), 3<->4 (Both, SRF), 4<->1
    // (SRF, Dry). Entries [1,3], [3,1], [2,4], [4,2] stay zero.
    array[N_classes, N_age_groups] matrix[4, 4] Q;
    array[N_classes] vector[4] log_alpha_init;
    vector[N_classes] log_theta = log(theta_class);

    for (c in 1:N_classes) {
      for (a in 1:N_age_groups) {
        vector[8] gamma_eff = (a == 1) ? rep_vector(0.0, 8) : gamma_age_fluid[a - 1];
        Q[c, a] = rep_matrix(0.0, 4, 4);
        Q[c, a, 1, 2] = exp(beta_fluid[c, 1] + gamma_eff[1]);   // Dry  -> IRF
        Q[c, a, 2, 1] = exp(beta_fluid[c, 2] + gamma_eff[2]);   // IRF  -> Dry
        Q[c, a, 2, 3] = exp(beta_fluid[c, 3] + gamma_eff[3]);   // IRF  -> Both
        Q[c, a, 3, 2] = exp(beta_fluid[c, 4] + gamma_eff[4]);   // Both -> IRF
        Q[c, a, 3, 4] = exp(beta_fluid[c, 5] + gamma_eff[5]);   // Both -> SRF
        Q[c, a, 4, 3] = exp(beta_fluid[c, 6] + gamma_eff[6]);   // SRF  -> Both
        Q[c, a, 4, 1] = exp(beta_fluid[c, 7] + gamma_eff[7]);   // SRF  -> Dry
        Q[c, a, 1, 4] = exp(beta_fluid[c, 8] + gamma_eff[8]);   // Dry  -> SRF
        for (k in 1:4) {
          Q[c, a, k, k] = -sum(Q[c, a, k, ]);                  // rows sum to 0
        }
      }
      log_alpha_init[c] = log(pi_init[c]);
    }

    // ---- 3. Transition matrices, once per (class, age group, unique gap) ---
    // The floor at 1e-12 guards log(0) from numerical underflow in expm.
    array[N_classes, N_age_groups, N_unique_dt] matrix[4, 4] log_P_unique;
    for (c in 1:N_classes) {
      for (a in 1:N_age_groups) {
        for (u in 1:N_unique_dt) {
          matrix[4, 4] P = matrix_exp(Q[c, a] * unique_dt[u]);
          for (r in 1:4) {
            for (s in 1:4) {
              log_P_unique[c, a, u, r, s] = log(fmax(P[r, s], 1e-12));
            }
          }
        }
      }
    }

    // ---- 4. Forward recursion in log space, per patient and class ---------
    // alpha_t(j) = sum_i alpha_{t-1}(i) P_ij(dt) * e_t(j); the class mixture is
    // taken outside the recursion: p(y_p) = sum_c theta_c p(y_p | c).
    for (p in 1:N_pat) {
      int a = pat_age_group[p];
      for (c in 1:N_classes) {
        vector[4] log_alpha = log_alpha_init[c];
        for (i in start_idx[p]:end_idx[p]) {
          if (dt_idx[i] > 0) {
            matrix[4, 4] log_P = log_P_unique[c, a, dt_idx[i]];
            vector[4] prev = log_alpha;
            for (j in 1:4) {
              log_alpha[j] = log_sum_exp(prev + log_P[, j]);
            }
          }
          log_alpha += log_em_cache[i]';
        }
        log_mix_pat[p, c] = log_theta[c] + log_sum_exp(log_alpha);
      }
      log_lik[p] = log_sum_exp(log_mix_pat[p]);
    }
  }
}

model {
  // ---- priors ---------------------------------------------------------------
  theta_class ~ dirichlet(rep_vector(1.0, N_classes));
  for (c in 1:N_classes) {
    pi_init[c] ~ dirichlet(rep_vector(1.0, 4));
  }

  // Observation error anchored at roughly 12% false-positive and 88%
  // true-positive rates; the anchor, together with the ordering constraint,
  // separates reading error from genuine transitions.
  theta_srf_logit[1] ~ normal(-2.0, 1.0);
  theta_srf_logit[2] ~ normal( 2.0, 1.0);
  theta_irf_logit[1] ~ normal(-2.0, 1.0);
  theta_irf_logit[2] ~ normal( 2.0, 1.0);

  // Weakly informative on log-intensities (per week) and on age effects.
  to_vector(beta_fluid) ~ normal(-2.5, 1.5);
  for (a in 1:(N_age_groups - 1)) {
    gamma_age_fluid[a] ~ normal(0.0, 1.0);
  }

  // ---- likelihood -----------------------------------------------------------
  target += sum(log_lik);
}

generated quantities {
  // Posterior class membership of each patient given the drawn parameters.
  array[N_pat] simplex[N_classes] class_post;
  for (p in 1:N_pat) {
    class_post[p] = softmax(log_mix_pat[p]');
  }
}

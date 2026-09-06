// =============================================================================
// finland_deploy_hmm_fast_nogap.stan
// Phase 2 of the two-phase prior-transfer prototype: the same latent-class
// CT-HMM fitted to the routine-care source (HUS doctoral study cohort), with
// informative priors imported from the Phase 1 trial fit and an absorbing
// drop-out state. "nogap" = the treat-and-extend visit-scheduling component of
// the full routine-care model is ablated; visit times are treated as given.
//
// What is transferred from Phase 1 (see hmm_transfer_pipeline.R)
//   beta_fluid[c, k]  ~ normal(prior_beta_fluid_mu[c, k], prior_beta_fluid_sd[c, k])
//   gamma_age_fluid   ~ normal(prior_gamma_fluid_mu, prior_gamma_fluid_sd)
//   pi_init[c]        ~ dirichlet(prior_pi_init_alpha[c])
//   where the mu/sd are elementwise means and standard deviations of the
//   label-aligned Phase 1 posterior. Priors are independent across elements:
//   posterior correlation is discarded and the prior widths are fixed, so the
//   strength of borrowing is set by construction and is not a parameter. This
//   is the limitation Article II of the research plan addresses.
//
// What is re-estimated without transfer
//   theta_class            class prevalence may differ between sources.
//   theta_*_logit_fin      reading error is specific to the routine-care
//                          imaging and readers.
//   beta_dropout, gamma_age_dropout   drop-out exists only in routine care.
//
// Class identity across sources is carried only by the transferred priors on
// beta_fluid: class c in this fit is "the class whose intensities are near
// Phase 1's aligned class c". Nothing in the parameterisation enforces it.
//
// Drop-out
//   State 5 is absorbing and unobserved except through the indicator
//   is_dropout, which the pipeline sets on a patient's terminal visit when
//   follow-up ended for a recorded reason. At that visit the emission puts all
//   mass on state 5 and the fluid readings of that visit are not used; the
//   transition into state 5 is therefore dated within the last observed
//   interval. Patients without the indicator are right-censored at their last
//   visit: state 5 is excluded at every observed visit, which is what makes
//   the drop-out process informative rather than ignorable.
//
// Outputs: log_lik[p], log_mix_pat[p, ], class_post[p] as in Phase 1.
//
// Status: exploratory prototype (September 2026).
// =============================================================================

data {
  // ---- observations, stored patient-contiguously ---------------------------
  int<lower=1> N_obs;
  int<lower=1> N_pat;
  array[N_pat] int<lower=1, upper=N_obs> start_idx;
  array[N_pat] int<lower=1, upper=N_obs> end_idx;
  int<lower=1> N_classes;

  // ---- time gaps, indexed for matrix-exponential caching -------------------
  int<lower=1> N_unique_dt;
  vector<lower=0>[N_unique_dt] unique_dt;
  array[N_obs] int<lower=0, upper=N_unique_dt> dt_idx;

  // ---- marker readings and drop-out indicator ------------------------------
  array[N_obs] int<lower=-1, upper=1> irf;
  array[N_obs] int<lower=-1, upper=1> srf;
  array[N_obs] int<lower=0, upper=1> is_dropout;   // 1 only on a terminal visit

  // ---- patient-level covariate: age group (1 = reference) -----------------
  int<lower=1> N_age_groups;
  array[N_pat] int<lower=1, upper=N_age_groups> pat_age_group;

  // ---- priors imported from the aligned Phase 1 posterior ------------------
  matrix[N_classes, 8] prior_beta_fluid_mu;
  matrix<lower=0>[N_classes, 8] prior_beta_fluid_sd;
  matrix[N_age_groups - 1, 8] prior_gamma_fluid_mu;
  matrix<lower=0>[N_age_groups - 1, 8] prior_gamma_fluid_sd;
  array[N_classes] vector<lower=0>[4] prior_pi_init_alpha;
}

transformed data {
  for (p in 1:N_pat) {
    if (start_idx[p] > end_idx[p])
      reject("patient ", p, ": start_idx exceeds end_idx");
    if (dt_idx[start_idx[p]] != 0)
      reject("patient ", p, ": first visit must have dt_idx == 0");
    if (is_dropout[start_idx[p]] == 1)
      reject("patient ", p, ": drop-out on the first visit gives zero likelihood");
    for (i in start_idx[p]:end_idx[p]) {
      if (is_dropout[i] == 1 && i != end_idx[p])
        reject("patient ", p, ": is_dropout set on a non-terminal visit");
    }
  }
  for (u in 1:N_unique_dt) {
    if (unique_dt[u] <= 0)
      reject("unique_dt[", u, "] must be positive");
  }
}

parameters {
  simplex[N_classes] theta_class;

  // Routine-care reading error, re-estimated (not transferred).
  ordered[2] theta_srf_logit_fin;
  ordered[2] theta_irf_logit_fin;

  matrix[N_classes, 8] beta_fluid;             // transferred prior
  matrix[N_classes, 4] beta_dropout;           // log drop-out intensity from
                                               // each fluid state, per class
  array[N_age_groups - 1] vector[8] gamma_age_fluid;     // transferred prior
  array[N_age_groups - 1] vector[4] gamma_age_dropout;

  array[N_classes] simplex[4] pi_init;         // transferred prior
}

transformed parameters {
  vector[N_pat] log_lik;
  matrix[N_pat, N_classes] log_mix_pat;
  {
    // ---- 1. Emission log-probabilities over five states ---------------------
    // Column 5 is the absorbing drop-out state. An observed visit excludes
    // state 5; a drop-out visit excludes states 1-4 and ignores its readings.
    matrix[N_obs, 5] log_em_cache = rep_matrix(0.0, N_obs, 5);
    for (i in 1:N_obs) {
      if (is_dropout[i] == 1) {
        log_em_cache[i, 1:4] = rep_row_vector(negative_infinity(), 4);
        log_em_cache[i, 5] = 0.0;
      } else {
        log_em_cache[i, 5] = negative_infinity();
        if (srf[i] != -1) {
          real lp_srf_absent  = bernoulli_logit_lpmf(srf[i] | theta_srf_logit_fin[1]);
          real lp_srf_present = bernoulli_logit_lpmf(srf[i] | theta_srf_logit_fin[2]);
          log_em_cache[i, 1] += lp_srf_absent;
          log_em_cache[i, 2] += lp_srf_absent;
          log_em_cache[i, 3] += lp_srf_present;
          log_em_cache[i, 4] += lp_srf_present;
        }
        if (irf[i] != -1) {
          real lp_irf_absent  = bernoulli_logit_lpmf(irf[i] | theta_irf_logit_fin[1]);
          real lp_irf_present = bernoulli_logit_lpmf(irf[i] | theta_irf_logit_fin[2]);
          log_em_cache[i, 1] += lp_irf_absent;
          log_em_cache[i, 2] += lp_irf_present;
          log_em_cache[i, 3] += lp_irf_present;
          log_em_cache[i, 4] += lp_irf_absent;
        }
      }
    }

    // ---- 2. Generator matrices per (class, age group), 5 x 5 --------------
    // Fluid ring as in Phase 1, plus a drop-out intensity from each fluid
    // state into state 5. Row 5 is zero: absorbing.
    array[N_classes, N_age_groups] matrix[5, 5] Q;
    array[N_classes] vector[5] log_alpha_init;
    vector[N_classes] log_theta = log(theta_class);

    for (c in 1:N_classes) {
      for (a in 1:N_age_groups) {
        vector[8] gamma_eff      = (a == 1) ? rep_vector(0.0, 8) : gamma_age_fluid[a - 1];
        vector[4] gamma_drop_eff = (a == 1) ? rep_vector(0.0, 4) : gamma_age_dropout[a - 1];
        Q[c, a] = rep_matrix(0.0, 5, 5);
        Q[c, a, 1, 2] = exp(beta_fluid[c, 1] + gamma_eff[1]);   // Dry  -> IRF
        Q[c, a, 2, 1] = exp(beta_fluid[c, 2] + gamma_eff[2]);   // IRF  -> Dry
        Q[c, a, 2, 3] = exp(beta_fluid[c, 3] + gamma_eff[3]);   // IRF  -> Both
        Q[c, a, 3, 2] = exp(beta_fluid[c, 4] + gamma_eff[4]);   // Both -> IRF
        Q[c, a, 3, 4] = exp(beta_fluid[c, 5] + gamma_eff[5]);   // Both -> SRF
        Q[c, a, 4, 3] = exp(beta_fluid[c, 6] + gamma_eff[6]);   // SRF  -> Both
        Q[c, a, 4, 1] = exp(beta_fluid[c, 7] + gamma_eff[7]);   // SRF  -> Dry
        Q[c, a, 1, 4] = exp(beta_fluid[c, 8] + gamma_eff[8]);   // Dry  -> SRF
        Q[c, a, 1, 5] = exp(beta_dropout[c, 1] + gamma_drop_eff[1]);   // Dry  -> out
        Q[c, a, 2, 5] = exp(beta_dropout[c, 2] + gamma_drop_eff[2]);   // IRF  -> out
        Q[c, a, 3, 5] = exp(beta_dropout[c, 3] + gamma_drop_eff[3]);   // Both -> out
        Q[c, a, 4, 5] = exp(beta_dropout[c, 4] + gamma_drop_eff[4]);   // SRF  -> out
        for (k in 1:4) {
          Q[c, a, k, k] = -sum(Q[c, a, k, ]);
        }
      }
      log_alpha_init[c] = rep_vector(negative_infinity(), 5);   // never start absorbed
      for (k in 1:4) {
        log_alpha_init[c, k] = log(pi_init[c, k]);
      }
    }

    // ---- 3. Transition matrices, once per (class, age group, unique gap) ---
    array[N_classes, N_age_groups, N_unique_dt] matrix[5, 5] log_P_unique;
    for (c in 1:N_classes) {
      for (a in 1:N_age_groups) {
        for (u in 1:N_unique_dt) {
          matrix[5, 5] P = matrix_exp(Q[c, a] * unique_dt[u]);
          for (r in 1:5) {
            for (s in 1:5) {
              log_P_unique[c, a, u, r, s] = log(fmax(P[r, s], 1e-12));
            }
          }
        }
      }
    }

    // ---- 4. Forward recursion in log space ----------------------------------
    for (p in 1:N_pat) {
      int a = pat_age_group[p];
      for (c in 1:N_classes) {
        vector[5] log_alpha = log_alpha_init[c];
        for (i in start_idx[p]:end_idx[p]) {
          if (dt_idx[i] > 0) {
            matrix[5, 5] log_P = log_P_unique[c, a, dt_idx[i]];
            vector[5] prev = log_alpha;
            for (j in 1:5) {
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

  // Transferred from Phase 1: independent normals on each log-intensity and
  // age effect, Dirichlet on each class's initial-state distribution.
  to_vector(beta_fluid) ~ normal(to_vector(prior_beta_fluid_mu),
                                 to_vector(prior_beta_fluid_sd));
  for (a in 1:(N_age_groups - 1)) {
    gamma_age_fluid[a] ~ normal(prior_gamma_fluid_mu[a]', prior_gamma_fluid_sd[a]');
  }
  for (c in 1:N_classes) {
    pi_init[c] ~ dirichlet(prior_pi_init_alpha[c]);
  }

  // Routine-care-only parameters: drop-out rare a priori, age effects weak.
  to_vector(beta_dropout) ~ normal(-4.0, 2.0);
  for (a in 1:(N_age_groups - 1)) {
    gamma_age_dropout[a] ~ normal(0.0, 1.0);
  }

  // Reading error, re-estimated with the same anchor as in Phase 1.
  theta_srf_logit_fin[1] ~ normal(-2.0, 1.0);
  theta_srf_logit_fin[2] ~ normal( 2.0, 1.0);
  theta_irf_logit_fin[1] ~ normal(-2.0, 1.0);
  theta_irf_logit_fin[2] ~ normal( 2.0, 1.0);

  // ---- likelihood -----------------------------------------------------------
  target += sum(log_lik);
}

generated quantities {
  array[N_pat] simplex[N_classes] class_post;
  for (p in 1:N_pat) {
    class_post[p] = softmax(log_mix_pat[p]');
  }
}

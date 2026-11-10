// Adopted H0/HA/HB working likelihood. No outcome fitting has occurred until
// a verified sampler actually returns retained posterior draws and diagnostics.
data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> D;
  int<lower=1> Q;
  int<lower=1> H;
  int<lower=1> F;
  array[F] int<lower=1, upper=3> interaction_order;
  array[J] int<lower=1, upper=F> family;
  matrix[N,J] X;
  matrix[N,H] slopes;
  vector[N] chance_logit;
  array[N] int<lower=1,upper=D> day;
  array[N] int<lower=1,upper=Q> query;
  array[N] int<lower=0,upper=1> hit;
  vector<lower=0>[N] likelihood_weight;
  real<lower=0> prior_scale;
}
parameters {
  real alpha;
  vector[J] z_beta;
  vector<lower=0>[F] tau;
  vector[D] z_day;
  vector[Q] z_query;
  matrix[D,H] z_slope;
  real<lower=0> sigma_day;
  real<lower=0> sigma_query;
  vector<lower=0>[H] sigma_slope;
}
transformed parameters {
  vector[J] beta;
  vector[N] eta;
  for (j in 1:J) {
    if (interaction_order[family[j]] == 1)
      beta[j] = prior_scale * 0.75 * z_beta[j];
    else
      beta[j] = prior_scale * tau[family[j]] * z_beta[j];
  }
  eta = chance_logit + rep_vector(alpha,N) + X*beta;
  for (i in 1:N)
    eta[i] += sigma_day*z_day[day[i]] + sigma_query*z_query[query[i]]
      + dot_product(slopes[i],z_slope[day[i]] .* sigma_slope');
}
model {
  alpha ~ normal(0,1.5);
  z_beta ~ std_normal();
  // Main-effect tau parameters are unused in beta but proper, fixed ancillary priors.
  for (f in 1:F) {
    if (interaction_order[f] == 3) tau[f] ~ normal(0,0.25);
    else tau[f] ~ normal(0,0.5);
  }
  z_day ~ std_normal();
  z_query ~ std_normal();
  to_vector(z_slope) ~ std_normal();
  sigma_day ~ normal(0,1);
  sigma_query ~ normal(0,1);
  sigma_slope ~ normal(0,0.5);
  for (i in 1:N)
    target += likelihood_weight[i]*bernoulli_logit_lpmf(hit[i] | eta[i]);
}
generated quantities {
  vector[N] probability_conditional = inv_logit(eta);
}

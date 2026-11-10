// HA W3 adopted two-way candidate. Original Helmert columns, no third order.
data {
  int<lower=1> N; int<lower=1> J; int<lower=1> F;
  int<lower=1> C; int<lower=1> D; int<lower=1> Q; int<lower=1> H;
  array[J] int<lower=0,upper=F> family; // zero: fixed-scale main effect
  matrix[C,J] X_cell;
  array[N] int<lower=1,upper=C> cell;
  array[N] int<lower=1,upper=D> day;
  array[N] int<lower=1,upper=Q> query;
  matrix[N,H] slopes;
  vector[N] chance_offset;
  array[N] int<lower=0,upper=1> hit;
  vector<lower=0>[N] weight;
  real<lower=0> prior_scale;
}
parameters {
  real alpha;
  vector[J] z_beta;
  vector<lower=0>[F] tau;
  vector[Q] z_query;
  vector[D] z_day;
  matrix[D,H] z_slope;
  real<lower=0> sigma_query;
  real<lower=0> sigma_day;
  vector<lower=0>[H] sigma_slope;
}
transformed parameters {
  vector[J] beta;
  for (j in 1:J) {
    if (family[j]==0) beta[j]=prior_scale*.75*z_beta[j];
    else beta[j]=prior_scale*tau[family[j]]*z_beta[j];
  }
}
model {
  vector[C] cell_eta=X_cell*beta;
  matrix[D,H] day_slopes=diag_post_multiply(z_slope,sigma_slope);
  vector[N] eta=chance_offset+rep_vector(alpha,N)+cell_eta[cell]
    +sigma_query*z_query[query]+sigma_day*z_day[day]
    +rows_dot_product(slopes,day_slopes[day]);
  alpha~normal(0,1.5);
  z_beta~std_normal(); tau~normal(0,.5);
  z_query~std_normal(); z_day~std_normal(); to_vector(z_slope)~std_normal();
  sigma_query~normal(0,1); sigma_day~normal(0,1); sigma_slope~normal(0,.5);
  // Exact power likelihood, including integer day multiplicity and zero holdout.
  for (i in 1:N) target+=weight[i]*bernoulli_logit_lpmf(hit[i] | eta[i]);
}

// Exact linear reparameterization of the adopted model. All coefficients and
// original Gaussian random effects are retained. T is fixed from design/chance,
// never outcomes. Householder transforms retain iid query Gaussian priors.
data {
  int<lower=1> N; int<lower=1> J; int<lower=1> C; int<lower=1> D;
  int<lower=1> Q; int<lower=0> H; int<lower=1> DS; int<lower=1> F;
  int<lower=0> CL; int<lower=1> NE; int<lower=1> G; int<lower=1> B;
  array[F] int<lower=1,upper=3> interaction_order;
  array[J] int<lower=1,upper=F> family;
  matrix[C,J] X_cell;
  matrix[DS,H] slope_design;
  array[DS] int<lower=1,upper=D> slope_day;
  array[N] int<lower=1,upper=NE> event;
  array[NE] int<lower=1,upper=C> event_cell;
  array[NE] int<lower=1,upper=D> event_day;
  array[NE] int<lower=1,upper=DS> event_slope;
  array[NE] int<lower=0,upper=CL> event_clip;
  vector[NE] event_offset;
  matrix[NE,2] event_basis;
  array[N] int<lower=1,upper=Q> query;
  array[N] int<lower=0,upper=1> hit;
  vector<lower=0>[N] likelihood_weight;
  real<lower=0> prior_scale;
  int<lower=0,upper=1> flexible_baseline;
  matrix[B,B] block_transform;
  real transform_log_abs_det;
  array[Q] int<lower=1,upper=G> query_group;
  array[Q] int<lower=0,upper=Q-G> query_deviation_index;
  vector[Q] householder_v;
}
transformed data {
  vector[N] y;
  for (i in 1:N) y[i]=hit[i];
}
parameters {
  vector[B] z_block;
  vector[Q-G] z_query_deviation;
  vector<lower=0>[F] tau;
  real<lower=0> sigma_day;
  real<lower=0> sigma_query;
  vector<lower=0>[H] sigma_slope;
  real<lower=0> sigma_clip;
}
model {
  vector[B] original_block=block_transform*z_block;
  real alpha=1.5*original_block[1];
  vector[J] z_beta=segment(original_block,2,J);
  vector[D] z_day=segment(original_block,J+2,D);
  matrix[D,H] z_slope=to_matrix(segment(original_block,J+2+D,D*H),D,H);
  vector[G] query_mean=segment(original_block,J+2+D+D*H,G);
  vector[CL] z_clip=segment(original_block,J+2+D+D*H+G,CL);
  vector[2] baseline_delta=segment(original_block,J+2+D+D*H+G+CL,2)
    .* [0.5,0.25]';
  vector[Q] query_temp;
  vector[G] projection=rep_vector(0,G);
  vector[Q] z_query;
  vector[J] beta;
  vector[C] cell_eta;
  matrix[D,H] b_day;
  vector[DS] slope_eta;
  vector[D] day_mu=sigma_day*z_day;
  vector[Q] query_mu;
  vector[NE] event_eta;
  vector[N] eta;
  for (q in 1:Q) {
    if (query_deviation_index[q]==0) query_temp[q]=query_mean[query_group[q]];
    else query_temp[q]=z_query_deviation[query_deviation_index[q]];
    projection[query_group[q]]+=householder_v[q]*query_temp[q];
  }
  for (q in 1:Q)
    z_query[q]=query_temp[q]-2*householder_v[q]*projection[query_group[q]];
  for (j in 1:J) {
    if (interaction_order[family[j]]==1) beta[j]=prior_scale*.75*z_beta[j];
    else beta[j]=prior_scale*tau[family[j]]*z_beta[j];
  }
  cell_eta=X_cell*beta;
  b_day=diag_post_multiply(z_slope,sigma_slope);
  slope_eta=rows_dot_product(slope_design,b_day[slope_day]);
  query_mu=sigma_query*z_query;
  event_eta=event_offset+rep_vector(alpha,NE)+cell_eta[event_cell]
    +day_mu[event_day]+slope_eta[event_slope];
  if (CL>0) event_eta+=sigma_clip*z_clip[event_clip];
  if (flexible_baseline==1) event_eta+=event_basis*baseline_delta;
  eta=event_eta[event]+query_mu[query];
  // Gaussian prior in ORIGINAL coordinates plus the fixed transformation Jacobian.
  // This is not a design-dependent change to the original priors.
  target+=std_normal_lpdf(original_block)+transform_log_abs_det;
  z_query_deviation~std_normal();
  for (f in 1:F) {
    if (interaction_order[f]==3) tau[f]~normal(0,.25);
    else tau[f]~normal(0,.5);
  }
  sigma_day~normal(0,1);sigma_query~normal(0,1);sigma_slope~normal(0,.5);
  sigma_clip~normal(0,1);
  if (min(likelihood_weight)==1 && max(likelihood_weight)==1)
    target+=bernoulli_logit_lpmf(hit | eta);
  else target+=dot_product(likelihood_weight,y .* eta-log1p_exp(eta));
}

"""
Multilevel models are regression models in which the constituent model parameters are given probability models.
This implies that model parameters are allowed to vary by group.

Observational units are often naturally clustered. Clustering induces dependence between observations,
despite random sampling of clusters and random sampling within clusters.

A hierarchical model is a particular multilevel model where parameters are nested within one another.

Some multilevel structures are not hierarchical.

e.g. "country" and "year" are not nested, but may represent separate, but overlapping, clusters of parameters
We will motivate this topic using an environmental epidemiology example.

Example: Radon contamination (Gelman and Hill 2006)
Radon is a radioactive gas that enters homes through contact points with the ground.
It is a carcinogen that is the primary cause of lung cancer in non-smokers.
Radon levels vary greatly from household to household.

The EPA did a study of radon levels in 80,000 houses. Two important predictors:
    - measurement in basement or first floor (radon higher in basements)
    - county uranium level (positive correlation with radon levels)

We will focus on modeling radon levels in Minnesota.
The hierarchy in this example is households within county.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_context('notebook')

# Import radon data
srrs2 = pd.read_csv('../data/srrs2.dat')  # household level data
srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2.assign(fips=srrs2.stfips*1000 + srrs2.cntyfips)[srrs2.state == 'MN']  # Minnesota only

cty = pd.read_csv('../data/cty.dat')   # county level data
cty_mn = cty[cty.st == 'MN'].copy()
cty_mn['fips'] = 1000*cty_mn.stfips + cty_mn.ctfips  # County level's predictor: Uranium (combine 2 variables)

# merge into 1 dataset - household level
srrs_mn = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on='fips')
srrs_mn = srrs_mn.drop_duplicates(subset='idnum')
u = np.log(srrs_mn.Uppm)  # predictor at county level
n = len(srrs_mn)

# need a dict for each unique county, for indexing
srrs_mn.county = srrs_mn.county.str.strip()
mn_counties = srrs_mn.county.unique()
counties = len(mn_counties)

county_lookup = dict(zip(mn_counties, range(len(mn_counties))))  # map counties' names to 0..J
county = srrs_mn['county_code'] = srrs_mn.county.replace(county_lookup).values
radon = srrs_mn.activity   # response variable
srrs_mn['log_radon'] = log_radon = np.log(radon + 0.1).values
floor_measure = srrs_mn.floor.values

# plot the response
log_radon_plt = srrs_mn.activity.apply(lambda x: np.log(x+0.1)).hist(bins=25)
radon_plt = srrs_mn.activity.hist(bins=25)
plt.show()

# Conventional approaches
# The 2 conventional alternatives to modeling radon exposure represent the two extremes of the bias-variance tradeoff:
#
# Complete pooling: Treat all counties the same, and estimate a single radon level. - max bias, low variance
# $$y_i = \alpha + \beta * x_i + \epsilon_i$$
#
# No pooling:  Model radon in each county independently.  - low bias - max variance because of few data points
# $$y_i = \alpha_{j[i]} + \beta x_i + \epsilon_i$$
# where $j = 1..85$
#
# The errors $\epsilon_i$ may represent measurement error, temporal within-house variation, or variation
# among houses.
#
# To specify the complete pooling model in Stan, we begin by constructing the data block, which includes vectors
# of log-radon measurements (y) and floor measurement covariates (x), as well as the number of samples (N).
pooled_data = """
data {
  int<lower=0> N; 
  vector[N] x;
  vector[N] y;
}
"""

# Next we initialize our parameters, which in this case are the linear model coefficients and the normal
# scale parameter. Notice that sigma is constrained to be positive.
pooled_parameters = """
parameters {
  vector[2] beta;
  real<lower=0> sigma;
} 
"""
# Finally, we model the log-radon measurements as a normal sample with a mean that is a function of the
#  floor measurement.
pooled_model = """
model {
  y ~ normal(beta[1] + beta[2] * x, sigma);
}
"""
# We then pass the code, data, and parameters to the stan function. The sampling requires specifying how many
# iterations we want, and how many parallel chains to sample. Here, we will sample 2 chains of length 1000.
import pystan
pooled_data_dict = {'N': len(log_radon), 'x': floor_measure, 'y': log_radon}
pooled_fit = pystan.stan(model_code=pooled_data + pooled_parameters + pooled_model,
                         data=pooled_data_dict, iter=1000, chains=2)

# The sample can be extracted for plotting and summarization.
pooled_sample = pooled_fit.extract(permuted=True)
b0, b1 = pooled_sample['beta'].T.mean(axis=1)   # the betas
plt.scatter(srrs_mn.floor, np.log(srrs_mn.activity+0.1))
xvals = np.linspace(-0.2, 1.2)
plt.plot(xvals, b1*xvals+b0, 'r--')
plt.ylabel('Radon estimate (log)'); plt.xlabel('Floor'); plt.title('Complete Pool model')

plt.show()

# At the other end of the extreme, we can fit separate (independent) means for each county. The only things
# that are shared in this model are the coefficient for the basement measurement effect, and the standard
# deviation of the error.
unpooled_model = """data {
  int<lower=0> N; 
  int<lower=1,upper=85> county[N];
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[85] a;
  real beta;
  real<lower=0,upper=100> sigma;
} 
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] = beta * x[i] + a[county[i]];
}
model {
  y ~ normal(y_hat, sigma);
}"""
unpooled_data = {'N': len(log_radon), 'county': county+1,  # Stan counts starting at 1
                 'x': floor_measure,  'y': log_radon}
unpooled_fit = pystan.stan(model_code=unpooled_model, data=unpooled_data, iter=1000, chains=2)
unpooled_estimates = pd.Series(unpooled_fit['a'].mean(0), index=mn_counties)
unpooled_se = pd.Series(unpooled_fit['a'].std(0), index=mn_counties)

# We can plot the ordered estimates to identify counties with high radon levels:
order = unpooled_estimates.sort_values().index

plt.scatter(range(len(unpooled_estimates)), unpooled_estimates[order])
for i, m, se in zip(range(len(unpooled_estimates)), unpooled_estimates[order], unpooled_se[order]):
    plt.plot([i,i], [m-se, m+se], 'b-')
plt.xlim(-1,86); plt.ylim(-1,4)
plt.ylabel('Radon estimate');plt.xlabel('Ordered county');
plt.show()

# Here are visual comparisons between the pooled and unpooled estimates for a subset of counties representing
# a range of sample sizes.
sample_counties = ('LAC QUI PARLE', 'AITKIN', 'KOOCHICHING',
                   'DOUGLAS', 'CLAY', 'STEARNS', 'RAMSEY', 'ST LOUIS')

fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
axes = axes.ravel()
b1_unpooled = unpooled_fit['beta'].mean(0)
for i, c in enumerate(sample_counties):
    y = srrs_mn.log_radon[srrs_mn.county == c]
    x = srrs_mn.floor[srrs_mn.county == c]
    axes[i].scatter(x + np.random.randn(len(x)) * 0.01, y, alpha=0.4)

    # No pooling model
    b = unpooled_estimates[c]

    # Plot both models and data
    xvals = np.linspace(-0.2, 1.2)
    axes[i].plot(xvals, b1_unpooled * xvals + b,          label='Unpooled')
    axes[i].plot(xvals, b1 * xvals + b0, 'r--', label='Com. Pooled')
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['basement', 'floor'])
    axes[i].set_ylim(-1, 3)
    axes[i].set_title(c)
    if not i % 2:
        axes[i].set_ylabel('log radon level')
plt.legend()
plt.show()

# NEITHER of these models are satisfactory:
# if we are trying to identify high-radon counties, pooling is useless
# we do not trust extreme unpooled estimates produced by models using few observations
# (e.g. ATKIN, KOOCHICHING, DOUGLAS, LAC QUI PARLE)


# Multilevel and hierarchical models can come in as a solution to this problem.
# Complete Pooling: we imply that data are sampled from the same model. This ignores any variation
# among sampling units (other than sampling variance).
#
# No pooling: When we analyze data unpooled, we imply that they are sampled independently from
# separate models.
# At the opposite extreme from the pooled case, this approach claims that differences between sampling
# units are too large to combine them.
#
# Partial pooling - remedies of the two:
# In a hierarchical model, parameters are viewed as a sample from a population distribution of parameters.
# Thus, we view them as being neither entirely different or exactly the same. This is partial pooling.

# We can use PyStan to easily specify multilevel models, and fit them using Hamiltonian Monte Carlo.
# The simplest partial pooling model for the household radon dataset is one which simply estimates
# radon levels, without any predictors at any level. A partial pooling model represents a compromise
# between the pooled and unpooled extremes, approximately a weighted average (based on sample size) of
# the unpooled county estimates and the pooled estimates.

# SEE FORMULA

# Estimates for counties with smaller sample sizes will shrink towards the state-wide average.
# Estimates for counties with larger sample sizes will be closer to the unpooled county estimates.
partial_pooling = """
data {
  int<lower=0> N; 
  int<lower=1,upper=85> county[N];
  vector[N] y;
} 
parameters {
  vector[85] a;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- a[county[i]];
}
model {
  mu_a ~ normal(0, 1);
  a ~ normal (10 * mu_a, sigma_a);

  y ~ normal(y_hat, sigma_y);
}"""

partial_pool_data = {'N': len(log_radon), 'county': county+1, 'y': log_radon}  # stan count starts at 1
partial_pool_fit = pystan.stan(model_code=partial_pooling, data=partial_pool_data, iter=1000, chains=2)

sample_trace = partial_pool_fit['a']

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
samples, counties = sample_trace.shape
jitter = np.random.normal(scale=0.1, size=counties)

n_county = srrs_mn.groupby('county')['idnum'].count()
unpooled_means = srrs_mn.groupby('county')['log_radon'].mean()
unpooled_sd = srrs_mn.groupby('county')['log_radon'].std()
unpooled = pd.DataFrame({'n': n_county, 'm': unpooled_means, 'sd': unpooled_sd})
unpooled['se'] = unpooled.sd / np.sqrt(unpooled.n)

axes[0].plot(unpooled.n + jitter, unpooled.m, 'b.')
for j, row in zip(jitter, unpooled.iterrows()):
    name, dat = row
    axes[0].plot([dat.n + j, dat.n + j], [dat.m - dat.se, dat.m + dat.se], 'b-')
axes[0].set_xscale('log')
axes[0].hlines(sample_trace.mean(), 0.9, 100, linestyles='--')

samples, counties = sample_trace.shape
means = sample_trace.mean(axis=0)
sd = sample_trace.std(axis=0)
axes[1].scatter(n_county.values + jitter, means)
axes[1].set_xscale('log')
axes[1].set_xlim(1, 100)
axes[1].set_ylim(0, 3)
axes[1].hlines(sample_trace.mean(), 0.9, 100, linestyles='--')
for j, n, m, s in zip(jitter, n_county.values, means, sd):
    axes[1].plot([n + j] * 2, [m - s, m + s], 'b-')
plt.show()
# Notice the difference between the unpooled and partially-pooled estimates, particularly at smaller sample sizes.
# The former are both more extreme and more imprecise.

# Varying intercept model  # UPSCALE THE TECHNIQUE:
# This model allows intercepts to vary across county, according to a random effect.
varying_intercept = """data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> county[N];
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[J] a;
  real b;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] = a[county[i]] + x[i] * b;
}
model {
  sigma_a ~ uniform(0, 100);
  a ~ normal (mu_a, sigma_a);
  b ~ normal (0, 1);
  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
}
"""
varying_intercept_data = {'N': len(log_radon),
                          'J': len(n_county),
                          'county': county+1, # Stan counts starting at 1
                          'x': floor_measure,
                          'y': log_radon}

varying_intercept_fit = pystan.stan(model_code=varying_intercept, data=varying_intercept_data, iter=1000, chains=2)

a_sample = pd.DataFrame(varying_intercept_fit['a'])
import seaborn as sns
sns.set(style="ticks", palette="muted", color_codes=True)

# Plot the orbital period with horizontal boxes
plt.figure(figsize=(16, 6))
sns.boxplot(data=a_sample, whis=np.inf, color="c")
plt.show()
varying_intercept_fit.plot(pars=['sigma_a', 'b']);
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:13:02 2020

@author: korayyenal
"""

#import pystan
import numpy as np
import pandas as pd
import pystan
import matplotlib as plt

#importing
df = pd.read_csv('/Users/korayyenal/trend2.csv')

#cleaning data
df = df.drop(columns=['cc'])
df = df.dropna()

#adding an index column for countries
uq_countries = df.country.unique()
lencountries = len(uq_countries)
country_lookup = dict(zip(uq_countries, range(lencountries)))
country = df['country_code'] = df.country.replace(country_lookup).values


countries    = df.country_code.values
year       = df.year.values
#log scaling y variable, i.e. religiousness
religiousness = np.log(df.church2.values)
inequality   = df.gini_net.values
rgdpl      = df.rgdpl.values

# Provide data
model_data = {'N': len(religiousness),
               
               'J1': len(countries),
               'J2min': df.year.min(),
               'J2max': df.year.max(),
              
               'c1': countries+1,
               'c2': year,
               
               'x': inequality,
               'ctrl_v': rgdpl,
               'y': religiousness}

# Model1: uninformative B / Random effects model with diffuse priors 

model1_code = """
data {

  #length
  int<lower=0> N; 
  int<lower=0> J1;
  int<lower=0> J2min;
  int<lower=0> J2max;
  
  #country and year data
  int<lower=1,upper=J1> c1[N];
  int<lower=J2min,upper=J2max> c2[N]; //min and max year included
  
  vector[N] ctrl_v; //rgdpl  
  vector[N] x; //inequality
  vector[N] y; //religiousness
} 
parameters {
  vector[J1] countries;
  vector[J2max-J2min+1] year;
  
  real beta;
  real gamma;
  
  real<lower=0,upper=100> sigma_countries;
  real<lower=0,upper=100> sigma_year;
  real<lower=0,upper=100> sigma_y;
  
} 
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] = countries[c1[i]]  +  year[c2[i]-J2min+1]  +  x[i] * beta  +  ctrl_v[i] * gamma;

}
model {
  sigma_countries ~ uniform(0, 100);
  countries ~ normal (0, sigma_countries);
  
  sigma_year ~ uniform(0, 100);
  year ~ normal (0, sigma_year);

  beta ~ normal (0, 1);
  gamma ~ normal (0, 1);
  
  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
  
  
}
"""

model1_fit = pystan.stan(model_code=model1_code, data=model_data, iter=1000, chains=2)
model1_fit.plot()

results = model1_fit.extract(permuted=True)
beta    = pd.DataFrame(results['beta'])
print("beta:\t\t", beta.mean().values, "\t", beta.std().values)

# informative B / Random effects model

model2_code = """
data {
  int<lower=0> N; 
  int<lower=0> J1;
  int<lower=0> J2min;
  int<lower=0> J2max;
  
  int<lower=1,upper=J1> c1[N];
  int<lower=J2min,upper=J2max> c2[N];
  
  vector[N] ctrl_v;
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[J1] countries;
  vector[J2max-J2min+1] year;
  
  real beta;
  real gamma;
  
  real<lower=0,upper=100> sigma_countries;
  real<lower=0,upper=100> sigma_year;
  real<lower=0,upper=100> sigma_y;
  
} 
transformed parameters { 
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] = countries[c1[i]]  +  year[c2[i]-J2min+1]  +  x[i] * beta  +  ctrl_v[i] * gamma;


}
model {
  sigma_countries ~ uniform(0, 100);
  countries ~ normal (0, sigma_countries);
  
  sigma_year ~ uniform(0, 100);
  year ~ normal (0, sigma_year);

  beta ~ normal (0, 100); 
  gamma ~ normal (0.1, 0.01);
  
  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
}
"""
model2_fit = pystan.stan(model_code=model2_code, data=model_data, iter=1000, chains=2)
model2_fit.plot()

results2 = model2_fit.extract(permuted=True)
beta2    = pd.DataFrame(results2['beta'])
print("beta2:\t\t", beta2.mean().values, "\t", beta2.std().values)


'''
Results:

beta:[0.07668336][0.00580843]
beta2:[0.06839533][0.00544634]

Coefficients for mean values of betas in two models are below:
Beta1: 0.07668336
Beta2:   0.06839533
Difference: -10.8%

Diffuse prior is uninformative, when have a biased prior ~N(0.1, 0.01). In this case, I expect the data plays a more important role in the posterior.

Also, the result when beta distribution is ~N(0, 100) is as follows:
Mean: 0.07787207 (higher than beta mean)
Std: 0.00670704 (higher than beta mean)

This indicates when variance is increased in the prior, the beta values also increase.
'''

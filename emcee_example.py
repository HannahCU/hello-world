# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:57:10 2019

@author: hanna
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
plt.close("all")

"""
Example to understand MCMC with emcee 

fitting a line to data with error bars that are
not believable 

generative probabilistic model
first step: write down liklihood function
(probability of a dataset given the model parameters)

equivalent to describing generative procedure for the data

linear model is considered here
underestimated uncertainties by constant fractional amount
"""

#generate synthetic data set (not obtained by direct measure but applicable all the same)

#choose 'true' parameters
m_true = -0.9594
b_true = 4.294
f_true = 0.534

#Generate some synthetic data from the model

N = 50 
x = np.sort(10*np.random.rand(N)) #array size 50
yerr = 0.1+*np.random.rand(N) #trvially y error
y = m_true*x + b_true #linear line using variables defined above
y += np.abs(f_true*y) * np.random.randn(N) #not sure why this happens, randn produces std normal dist of shape array of N samples
y += yerr * np.random.randn(N) #not sure why either


def line(x, a, b):
    return a*x + b
popt, pcov = op.curve_fit(line, x, y, sigma=yerr)
plt.errorbar(x,y,yerr, fmt='none')
xfine = np.linspace(0,np.max(x),100)

#true value plot
plt.plot(xfine, m_true*xfine + b_true, 'b-', label="True fit")

#curvefit plot
plt.plot(xfine, line(xfine, popt[0], popt[1]), 'r-', label="Curve fit")
print("a = ", popt[0], "+/-", pcov[0,0]**0.5)
print("b = ", popt[1], "+/-", pcov[1,1]**0.5)


#fit a line using least linear squares
#assuming independent gaussian error bars
A = np.vstack((np.ones_like(x), x)).T      #stack vertically and transpose
C = np.diag((yerr * yerr))                 #covaiance matrix, values of error squared, diagonal through matrix
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C,A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

plt.plot(xfine, m_ls*xfine + b_ls, 'y', label="Least Square")

#Maximum likelihood estimation
#the least squares solution is the max likelihood result where
#error bars assumes correct, gaussian and independent
#if the model isn't right, there's no least square generalization
#we need to numerically optimize the likelihood function
# likelihood function is a gaussian with underestimated variance 
#code as follows 

def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y - model)**2*inv_sigma2 - np.log(inv_sigma2))) #should there be a 2pi in the log bracket surely



nll = lambda * args: -lnlike(*args) #lamda is used to define a function not using def, args is its variable, after colon is function
result = op.minimize(nll, [m_true, b_true, np.log(f_true)], args=(x,y,yerr)) #function, initial guesses, additional arguments passed to objective (nll) function
m_ml, b_ml, lnf_ml = result["x"] #this result["x"] gives m,b, lnf in one umbrella term
plt.plot(xfine, line(xfine, m_ml, b_ml), 'm-', label="Max Likelihood")
print(m_ml)
print(b_ml)
plt.title("Plot of Least Square and Minimized Likelihood Fits")
plt.legend()

#note scipy.optimize minimizes functions, we want to maximize the likelihood
#this is the same as negative likelihood (log likelihood in this case)
#we need to estimate the uncertainties of m and b 
#we should propagate uncertainties in f through 
#MCMC comes in here 


"""

common reason for use of MCMC is the want to marginalize over
some 'nuisance parameters' and find estimate of posterior probability 
function for others, MCMC does both in one.

start by writing down posterior probability function (up to a constant ie
proportionality). We have already written likelihood function, 
so just missing prior function. 

Priors are important! MCMC draws samples from a probability distribution
and you want this to be from your parameters.
YOU CANNOT DRAW PARAMETER SAMPLES FROM YOUR LIKLIHOOD FUNCTION!
(this is because its a probability distribution over datasets, so 
can draw conditional datasets but not parameter samples)

Here we consider uninformative (uniform) priors on m, b and lnf

    |1/5.5, if -5 < m < 1/2
p(m)|
    |0,     otherwise

"""

def lnprior(theta):            #log prior
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf           #whats going on here


#combine with lnlike from above to get full log probability function
    
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp): #false if infinity or not a number
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)    #sum log prior and log liklihood func, i.e log of posterior prob up to constant


#can sample this distribution using emcee now its all set up 
#begin - intialize the walkers in a tiny Gaussian ball around maximum 
#liklihood result (good initialization in most cases)
    
ndim, nwalkers = 3, 100         #no. dimensions, no. walkers
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] #think this is a list comprehension?

#pos is a list of length 100 with each entry being an array of length 3 themselves
#now set up sampler

import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

#run MCMC for 500 steps starting from the tiny ball defined above 

step = np.linspace(0, 500, 1)
sampler.run_mcmc(pos, 500)
#sampler object now has attribute 'chain' - an array shape (100,500,3) giving 
#parameter values for each walker at each step in the chain

#print(sampler.chain.shape)
#print(sampler.flatchain.shape)

print(sampler.chain[0:5,0,:])
fig, ax = plt.subplots(3,1, sharex = True) #conveniently packs fig ang subplots in 1, sharex = share x axes
for i in [0,1,2]:
    ax[i].plot(sampler.chain[0,:,i], 'k-', lw=0.2)
    ax[i].plot([0,499], [sampler.chain[0,:,i].mean(), sampler.chain[0,:,i].mean()], 'r-') #plots mean(true) in red
    plt.suptitle("Random walker position for m, b, lnf as a function of steps") #puts title at top

# note cant get middle plot to show as the y axis doesnt reach the data (y=4)   



"""What has happened? walkers start in small distributions around max liklihood
values and quickly begin to explore full posterior distribution
sample is well 'burnt in' after only 50 steps, ie steps taken to stabilise chain

for now we accept this and discard initial 50 steps and flatten chain to get a flat
list of samples 
"""

samples = sampler.chain[:, 50:, :].reshape((-1,ndim))


#corner plot
import corner 

fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"], 
                    truths = [m_true, b_true, np.log(f_true)])
fig.savefig("triange.png")

"""this shows all 1 and 2d dimension projections of posterior probability
distributions of your parameters. 
Demonstrates covariances between parameters (ie measures strength of correlation)

you find the marginalized distribution for a (set of) parameter(s) using the results
of the MCMC chain by projecting the samples into that plane and make
an N dimensional histogram.

Corner plot shows marginalized distribution of each parameter independently 
in the histograms along the diagonal 
and the marginalized 2D distributions in the other panels 

another diagnostic plot is projection of results into space of the observed data.
to do so:
    choose a few samples from the chain (100)
    plot ontop of the data points 
    
"""

xl = np.array([0,10])

plt.figure(4)
for m, b, lnf in samples[np.random.randint(len(samples), size=100)]: #if high isnt specified results are from 0 - low (ie len(samples))
    
    plt.plot(xl, m*xl+b, "k", alpha = 0.1)
plt.plot(xl, m_true * xl + b_true, "r", lw = 2, alpha = 0.8)
plt.errorbar(x, y , yerr, fmt=".k", capsize = 1, lw=0.5)
plt.title("Results plotted over observed data points")


"""
which numbers should go in the abstract?
could quote uncertainties based on 16th, 50th, 84th percentiles of the samples
in marginalized distributions """

samples[:, 2] = np.exp(samples[:,2])

m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2] -v[1], v[1]-v[0]), 
                             zip(*np.percentile(samples, [16,50,84], axis=0)))

print(m_mcmc)
print(b_mcmc)
print(f_mcmc)
#these values are close to the true values 












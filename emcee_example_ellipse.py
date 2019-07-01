# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:57:10 2019

@author: hanna
"""


""" This is an adaptation for an ellipse"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
plt.close("all")


u=0    #x-position of the center
v=0   #y-position of the center
a_true=2.     #radius x-axis
b_true=1.5    #radius y-axis
N = 100
t = np.sort(2*np.pi*np.random.rand(N))
yerr = -0.1 + 0.5*np.random.rand(len(t)) # y error
xerr = -0.25 + 0.5*np.random.rand(len(t)) # y error
x = u+a_true*np.cos(t) 
y = v+b_true*np.sin(t)
#y += yerr # * np.random.randn(len(t)) 
#x += xerr #* np.random.randn(len(t)) 


tfine = np.linspace(0,10,100)
plt.errorbar(x,y, yerr, xerr, fmt=".k", capsize = 1, lw=0.5)
plt.plot(u+a_true*np.cos(tfine), v+b_true*np.sin(tfine),"r", lw=3, label="True")
plt.legend()
plt.title("Plot of an Ellipse")

def lnlike(theta, t, xerr, yerr):
    a, b = theta
    modely = v+b*np.sin(t)
    x = u+a_true*np.cos(t) + xerr
    y = v+b_true*np.sin(t) + yerr
    inv_sigma2y = 1.0/(yerr**2)
    modelx = u+a*np.cos(t)
    inv_sigma2x = 1.0/(xerr**2)
    fy = -0.5*(np.sum((x - modelx)**2*inv_sigma2x - np.log(inv_sigma2x) + (y - modely)**2*inv_sigma2y - np.log(inv_sigma2y))) #should there be a 2pi in the log bracket surely
    return fy



nll = lambda * args : -lnlike(*args) 
result = op.minimize(nll, [a_true ,b_true], args=(t, xerr, yerr)) #function, initial guesses, additional arguments passed to objective (nll) function
a_ml, b_ml = result["x"]
plt.plot(u+a_ml*np.cos(tfine), v+b_ml*np.sin(tfine), 'b-', label="Max Likelihood")
print("Max likelihood a is", a_ml, "Max likelihood b is", b_ml)
plt.title("Plot of Maxmized Likelihood Fit")
plt.legend()







def lnprior(theta):            #log prior
    a, b = theta
    if 0 < a < 5 and 0.0 < b < 10.0:
        return 0.0
    return -np.inf           #whats going on here

#combine with lnlike from above to get full log probability function
def lnprob(theta, t, xerr, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp): #false if infinity or not a number
        return -np.inf
    return lp + lnlike(theta, t, xerr, yerr)    #sum log prior and log liklihood func, i.e log of posterior prob up to constant




ndim, nwalkers = 2, 8        #no. dimensions, no. walkers
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] #think this is a list comprehension?



import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, xerr, yerr))

#run MCMC for 500 steps starting from the tiny ball defined above 
iteration = 500
sampler.run_mcmc(pos, iteration)



print("Sampler chain is", sampler.chain[0,0,:])
fig, ax = plt.subplots(2,1, sharex = True) #conveniently packs fig ang subplots in 1, sharex = share x axes
for i in [0,1]: #note must specify all numbers here as no range specified
    ax[i].plot(sampler.chain[0,:,i], 'k-', lw=0.2)
    ax[i].plot([0,499], [sampler.chain[0,:,i].mean(), sampler.chain[0,:,i].mean()], 'r-') #plots mean(true) in red
    plt.suptitle("Random walker position for a, b as a function of steps") #puts title at top



samples = sampler.chain[:, 200:, :].reshape((-1,ndim))


#corner plot
import corner 
fig = corner.corner(samples, labels=["$a$", "$b$"], 
                    truths = [a_true, b_true])
fig.savefig("triange.png")




plt.figure(4)
for a, b in samples[np.random.randint(len(samples), size = 100)]: #if high isnt specified results are from 0 - low (ie len(samples))
    plt.plot(u+a*np.cos(tfine), v+b*np.sin(tfine), "b", lw=2, alpha = 0.3)
plt.plot(u+a_true*np.cos(tfine), v+b_true*np.sin(tfine), "r", lw = 2, alpha = 0.8)
plt.errorbar(x, y , yerr, fmt=".k", capsize = 1, lw=0.5)
plt.title("Results plotted over observed data points")

    











"""
which numbers should go in the abstract?
could quote uncertainties based on 16th, 50th, 84th percentiles of the samples
in marginalized distributions """

samples[:, 1] = np.exp(samples[:,1])

a_mcmc, b_mcmc = map(lambda v: (v[1], v[2] -v[1], v[1]-v[0]), 
                             zip(*np.percentile(samples, [16,50,84], axis=0)))
print(a_mcmc)
print(b_mcmc)
#these values are close to the true values 






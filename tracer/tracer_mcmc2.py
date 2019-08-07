import numpy as np
import numpy
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import scipy.misc as smp
import scipy.optimize as op
from mpl_toolkits import mplot3d
from scipy.integrate import odeint
from PIL import ImageFont
from PIL import ImageDraw

#importing raytracer function
from raytracer_optimized import raytracer

"""implement MCMC to raytracer
we want to loop over all pixels to get a large number
of data points
model is unnoisy image
observed data is noisy image
likelihood function can be a fitted gaussian distribution
so we have model - observed / error
we do this for each pixel
e.g. say observed is black (=0) and model is black (=0)
there is no difference """

"""
assign black pixels (0,0,0) is 0
and everything else as 1
now array is of form 200x200x1
note can flatten it from 200x200 array to 200x200 1d row
likelihood is fine otherwise though atm
then need to loop over every image and do this for every image
"""

truespin = 0.9
pixdensity = 6


#define sigma for likelihood
sigma = 0.2
#var = sigma**2
#function to add noise to image, this is now the observed data
def noisy(noise_typ,image,sigma):
   if noise_typ == "gauss":
      row,col= image.shape
      mean = 0
      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy =  -0.5 + image + gauss #add pos and neg error
      #print(gauss)
      return noisy

#creates observed data from 0.9 spin image
fornoise = raytracer(0.9, pixdensity)#Image.open("/Users/hannahriley/Desktop/raytracerpngs/spin0.9.png")
fornoisearray = np.asarray(fornoise)
noisearray = np.empty([6,6]) #needs fixing so these are automatically same shape as fornoisearray (10,10) but not (10,10,3)
#counter=0
#counter2=0
for i in range(len(fornoisearray)): #reshapes from 10,10,3 to 10,10 (ie sums the 3)
    #counter2+=1
    for j in range(len(fornoisearray[i,:])):
        #counter+=1
        noisearray[i,j]=np.sum(fornoisearray[i,j])
binfor = (noisearray!=0).astype(int)
obsarray = noisy("gauss",binfor,sigma)
obsim = smp.toimage(obsarray)
#obsim.show()
#obsarray.flatten()

#define likelihood function
def lnlike(spin, obs, pixdensity, sigma):
    #call raytracer function
    #np.reshape(obs, [6,6])
    array = raytracer(spin, pixdensity)
    array2 = np.empty([6,6]) #needs fixing so these are automatically same shape as fornoisearray (10,10) but not (10,10,3)
    #counter=0
    #counter2=0
    for i in range(len(array)): #reshapes from 10,10,3 to 10,10 (ie sums the 3)
        #counter2+=1
        for j in range(len(array[i,:])):
            #counter+=1
            array2[i,j]=np.sum(array[i,j])
    mod = (array2!=0).astype(int)
    lnlike = 0
    inv_sigma2 = 1.0/(sigma**2) #from noisy function, sigma is sqrt(var)
    newar = obs - mod
    for el in newar:
        fi = (el)**2*inv_sigma2
        fy = (-0.5*(np.sum((fi) - np.log(inv_sigma2))))
        lnlike = lnlike + fy
    return lnlike

#       MUST FIX!!!!
# nll = lambda * args : -lnlike(*args)
# result = op.minimize(nll, truespin, args=(obsarray, pixdensity, sigma)) #function, initial guesses, additional fixed arguments passed to objective (nll) function
# spinml = result["x"]


def lnprior(spin):            #log prior
    if 0.0 <= spin <= 1.0:     #Uninformative - range
        return 0.0
    return -np.inf

#combine with lnlike from above to get full log probability function
def lnprob(spin, obs, pixdensity, sigma):
    lp = lnprior(spin)
    if not np.isfinite(lp): #false if infinity or not a number
        return -np.inf
    return lp + lnlike(spin, obs, pixdensity, sigma)    #sum log prior and log liklihood func, i.e log of posterior prob up to constant

post = lnprob(truespin, obsarray, pixdensity, sigma)
print("posterior is", post)


#truespin in pos should be result from max likelihood but for now we use truespin til fixed
ndim, nwalkers = 1, 4        #no. dimensions, no. walkers
pos = [truespin + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]


import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(obsarray, pixdensity, sigma))

#run MCMC for 100 steps starting from the tiny ball defined above
iteration = 100
sampler.run_mcmc(pos, iteration)



print("Sampler chain is", sampler.chain[0,0,:])
fig, ax = plt.subplots(1,1, sharex = True) #conveniently packs fig ang subplots in 1, sharex = share x axes
for i in [0]: #note must specify all numbers here as no range specified
    ax[i].plot(sampler.chain[0,:,i], 'k-', lw=0.2)
    ax[i].plot([0,99], [sampler.chain[0,:,i].mean(), sampler.chain[0,:,i].mean()], 'r-') #plots mean(true) in red
    plt.suptitle("Random walker position for spin as a function of steps") #puts title at top



samples = sampler.chain[:, 25:, :].reshape((-1,ndim))


#corner plot
import corner
fig = corner.corner(samples, labels=["$spin$"],
                    truths = [spintruth])
#fig.savefig("triange.png")




# plt.figure(4)
# for spin in samples[np.random.randint(len(samples), size = 100)]: #if high isnt specified results are from 0 - low (ie len(samples))
#     plt.plot(u+a*np.cos(tfine), v+b*np.sin(tfine), "b", lw=2, alpha = 0.3)
# plt.plot(u+a_true*np.cos(tfine), v+b_true*np.sin(tfine), "r", lw = 2, alpha = 0.8)
# plt.errorbar(x, y , yerr, fmt=".k", capsize = 1, lw=0.5)
# plt.title("Results plotted over observed data points")

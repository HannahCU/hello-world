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
import random
import warnings
warnings.filterwarnings("ignore")

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



trueview = np.pi/4
truespin = 0.3
pixdensity = 10
sigma = 0.1 #define sigma for likelihood
#var = sigma**2




#function to add noise to image, this is now the observed data
def noisy(noise_typ,image,sigma):
    if noise_typ == "gauss":
        row = image.shape #for shape (10,10,3) need row,col,ch respectively to be defined
        print(row)
        mean = 0
        gauss = np.random.normal(mean,sigma,(row))
        gauss = gauss.reshape(row)
        noisy = image + gauss #add pos and neg error
        #print(gauss)
        return noisy, gauss


#creates observed data from 0.9 spin image
fornoise = raytracer(trueview, truespin, pixdensity)    #returns binary flat array of image(model) #Image.open("/Users/hannahriley/Desktop/raytracerpngs/spin0.9.png")
print(fornoise)
#fornoisearray = np.asarray(fornoise) #this isnt needed for raytracer func call as already in array form
#noisearray = np.empty([pixdensity**2]) #for 2D change back to [pixdensity,pixdensity]
#counter=0
#counter2=0
#for i in range(len(fornoise)): #reshapes from 10,10,3 to 10,10 (ie sums the 3)
    #counter2+=1
    #for j in range(len(fornoisearray[i,:])):
        #counter+=1
    #noisearray[i]=np.sum(fornoise[i]) #to return to 2d would need index [i,j]
#binfor = (noisearray!=0).astype(int) #already done in raytracer function
obsarray, gauss = noisy("gauss",fornoise,sigma) #was binfor originally
print("obs data:\n",obsarray)
print("inv_sigma2",1/gauss**2)

#obsarr_im = np.reshape(obsarray, [pixdensity,pixdensity])
#obsim = smp.toimage(obsarr_im)
#obsim.show()
#obsarray.flatten()

#define likelihood function
def lnlike(theta, obs, pixdensity, gauss):
    view, spin = theta
    model = raytracer(view, spin, pixdensity)
    lnlike = 0
    inv_sigma2 = 1.0/(gauss**2) #from noisy function, sigma is sqrt(var)
    diff = obs - model
    #print("diff\n", diff)
    lnlike = -0.5* np.sum(diff*diff * inv_sigma2)
    return lnlike

#       MUST FIX!!!!
# nll = lambda * args : -lnlike(*args)
# result = op.minimize(nll, truespin, args=(obsarray, pixdensity, sigma)) #function, initial guesses, additional fixed arguments passed to objective (nll) function
# spinml = result["x"]

def lnprior(theta):            #log prior
    view, spin = theta
    if 0 < view < np.pi/2 - 0.1 and 0.0 < spin < 0.999:
        return 0.0
    return -np.inf

#combine with lnlike from above to get full log probability function
def lnprob(theta, obs, pixdensity, sigma):
    lp = lnprior(theta)
    if not np.isfinite(lp): #false if infinity or not a number
        return -np.inf
    lpr = lp + lnlike(theta, obs, pixdensity, sigma)
    print(theta, lpr)
    return lpr    #sum log prior and log liklihood func, i.e log of posterior prob up to constant

#post = lnprob(truespin, obsarray, pixdensity, sigma)
#print("posterior is", post)


#truespin in pos should be result from max likelihood but for now we use truespin til fixed
ndim, nwalkers = 2, 4      #no. dimensions, no. walkers
pos =[(random.uniform(0,np.pi/2-0.1),random.uniform(0,0.999)) for i in range(nwalkers)] #[truespin + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
print("pos is", pos)


import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(obsarray, pixdensity, sigma))

#run MCMC for 100 steps starting from the tiny ball defined above
iteration = 200
sampler.run_mcmc(pos, iteration)



#missing save doc section as doesnt work


nburn = 100 # take out the first nburn iterations as for burn-in
sam = sampler.flatchain[:,:]
#print("Sampler chain is", sam)
plt.hist(sam[nburn:,0], color='b', bins=25, label='viewing angle') #ADDING the viewtruth virtical line
plt.legend(loc=0)
plt.xlabel("Viewing Angle")
plt.ylabel("Frequency")
plt.savefig("view.png")
plt.show()
plt.clf()

plt.hist(sam[nburn:,1], color='g', bins=25, label='spin') #ADDING the spintruth virtical line
plt.legend(loc=0)
plt.xlabel("Spin")
plt.ylabel("Frequency")
plt.savefig("spin.png")
plt.show()
plt.clf()

fig, ax = plt.subplots(2,1, sharex = True) #conveniently packs fig ang subplots in 1, sharex = share x axes

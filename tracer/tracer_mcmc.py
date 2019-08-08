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


def findpix(e):
    name = "/Users/hannahriley/Desktop/raytracerpngs/"
    name += str(e)
    im = Image.open(name, 'r')

    pix_val = list(im.getdata()) #scans horziontally from left to right from top left corner
    #pix_val_flat = [x for sets in pix_val for x in sets]
    lightpix = [pix_val[i] for i in range(len(pix_val)) if pix_val[i] != (0,0,0)]
    arearatio = len(lightpix)/len(pix_val) #ratio of light pixels to total pixels

    return im, pix_val, lightpix, arearatio

#define sigma for likelihood
#large to be visible on image
sigma = 0.2
var = sigma**2
#funcion to add noise to image, this is now the observed data
def noisy(noise_typ,image,sigma):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy =  -0.5 + image + gauss #add pos and neg error
      #print(gauss)
      return noisy




#define likelihood function
def lnlike(mod, obs, sigma):
    #fy = np.empty_like(mod)
    #call raytracer function
    lnlike = 0
    inv_sigma2 = 1.0/(sigma**2) #from noisy function, sigma is sqrt(var)
    newar = obs - mod
    for el in newar:
        fi = (el)**2*inv_sigma2
        fy = (-0.5*(np.sum((fi) - np.log(inv_sigma2))))
        lnlike = lnlike + fy
    return lnlike


def lnprior(spin):            #log prior
    if 0.0 <= spin <= 1.0:
        return 0.0
    return -np.inf

#combine with lnlike from above to get full log probability function
# def lnprob(spin, mod, obs, sigma):
#     lp = lnprior(spin)
#     if not np.isfinite(lp): #false if infinity or not a number
#         return -np.inf
#     return lp + lnlike(mod, obs, sigma)    #sum log prior and log liklihood func, i.e log of posterior prob up to constant



#creates observed data from 70deg image
fornoise = Image.open("/Users/hannahriley/Desktop/raytracerpngs/spin0.9.png")
fornoisearray = np.asarray(fornoise)
print(fornoisearray[0,:])
noisearray = np.empty([10,10]) #needs fixing so these are automatically same shape as fornoisearray (10,10) but not (10,10,3)
#counter=0
#counter2=0
for i in range(len(fornoisearray)): #reshapes from 10,10,3 to 10,10 (ie sums the 3)
    #counter2+=1
    for j in range(len(fornoisearray[i,:])):
        #counter+=1
        noisearray[i,j]=np.sum(fornoisearray[i,j])

# print(noisearray[0,:])
print(counter)
print(counter2)
# for x in range(3,8):
#     print(fornoisearray[:,x])
# print(noisearray)
binfor = (noisearray!=0).astype(int)
print(binfor)
obsarray = noisy("gauss",binfor,sigma)
obsim = smp.toimage(obsarray) #need to check the error added to this.doesnt seem right
#obsim.show()


fy=[]
#work out likelihood for each image against noisy model
for filename in os.listdir("/Users/hannahriley/Desktop/raytracerpngs/"):
    if filename.endswith('.png'):
        model, pix_val, lightpix, area = findpix(filename)

        #modelarray gives an array shape (200,200,3)
        #which is pixel dimensions 200x200 in this case. each pixel
        #has 3 coords denoting RGB values, e.g. (0,0,0) = black
        modelarray = np.asarray(model)
        if modelarray.shape == (10,10,4): #some reason one of the images is wrong shape so catch and ignore
            continue

        #assigned pixvals of (0,0,0) = 0, and anything above 0 as 1
        binmod = (modelarray!=0).astype(int)
        #print(binmod[100,100]) #shows that the pixvals are different for some of the images
        #print(area)

        #define observed data
        #obsarray = noisy("gauss",binmod)
        #obsim = smp.toimage(obsarray)
        # im.show()
        # obsim.show()

        fy.append(lnlike(binmod, obsarray, sigma))
    #    fp.append(lnprob(binmod,obsarray,sigma))

print("fy is:", fy)
#print(np.argmin(np.abs(fy))) #note these are negative LOG likelihoods, so very small positive likelihood
#print(np.exp(fy[9])) #of the order e-177
#print("lnprob", lnprob)

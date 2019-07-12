# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:13:07 2019

@author: hanna
"""


"""rantonel schwarzchild

solveode 

edited to produce my own schwarzchild ray tracer WITH accretion disk
"""

from matplotlib.patches import Wedge 
import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.close()



#useful funcs
def w(r):
    return (1 - 1/r) #term in metric?

def wu(u):
    return (1/u) # r = 1/u

#think the following is the equation of motion for photon trajectory
    #should make use of impact parameter i think (b=l/e)

def mUprime(u):
    return -0.5*(2*u - 3*u**2)  #2nd ODE geodesic eq, converted from r to u 

def func(u,t):



    # since we integrate over all phis, without stopping, THEN crop

    # the solution where it hits the EH or diverges, we don't want

    # it to wander aimlessly. We force it to stop by erasing the derivative.

    

    if (u[0]>1) or (u[0] < 0.0001):

        return [0,0]

    return [u[1], mUprime(u[0])]



def gradient(u,t): #is this necessary?


    #Jacobian of above


    return [[0,1],[1-3*u[0] , 0]]


def find_first(item, vec):
    #first occurrence 
    for l in range(len(vec)):
        if item >= vec[l]:
            return l
    return -1


# give a solution for one initial condition
# returns (array of phis, array of [u(phi), u'(phi)]).



def geod(r0,r0prime,options={}):



    u0 = [ 1/r0 , -r0prime/(r0*r0) ]





    if('step' in options):

        timestep = options['step']

    else:

        timestep = 0.005



    if('maxangle' in options):

        maxangle = options['maxangle']

    else:

        maxangle = 10



    phi = np.arange(np.pi/2,maxangle,timestep) #phi defined 



    u = odeint(func,u0,phi, Dfun=gradient, printmessg=False)

    

    return (phi,u)

# solves a list of initial condition and yields


r0 = 1.8 #allows to define initial radius
theta = 0.4 #initial angle of trajectory

# list of solutions in the format above.

xd = np.linspace(-2,2,100)
yd = np.zeros(len(xd))
def accdisk(m,xd,c):
    return m*xd + c
popt,pcov = curve_fit(accdisk, xd, yd)
#plt.plot(xd, accdisk(xd, popt[0], popt[1]), 'b-', label="Curve fit")

acdmin = -2 #disk limits
acdmax = -1



for i in range(700): #for many rays
    r0prime = -r0*np.tan(theta)+r0*1.5*np.tan(theta)*i/700
    phi, uup = geod(r0, r0prime)
    u = uup[:,0] #separate u from [u,uprime]
    uprime = uup[:,1]

    r = wu(u) #convert back to r
    x = np.array(r*np.cos(phi)) #convert from polar to cartesian 
    y = np.array(r*np.sin(phi))
    #yacc = y[y<0]
    
    a = find_first(0,y) #finds first instance where ray crosses horizontal 

    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)  
    ax.set_xlim([-400,400])
    ax.set_ylim([-400,400])
    #if 1<abs(x[a])<2:#plots on polar plot 
    tang = r0prime/r0
    z = 10/tang
        
    
    if 0.95<abs(x[a])<1.25:
        ring1 = Wedge((0,0), z, 0, 360, width=0.01, color='c')
        ax.add_patch(ring1)
        

    elif 1.26<abs(x[a])<1.5:
        ring1 = Wedge((0,0), z, 0, 360, width=0.01, color='y')
        ax.add_patch(ring1)

    elif 1.51<abs(x[a])<1.75:
        ring1 = Wedge((0,0), z, 0, 360, width=0.01, color= 'm')
        ax.add_patch(ring1)

    elif 1.76<abs(x[a])<2:
        ring1 = Wedge((0,0), z, 0, 360, width=0.01,color= 'b')
        ax.add_patch(ring1)
            
    plt.figure(2) #plots the accretion disk in diff colors 
    if acdmin<x[a]<-1.75:
        plt.plot(x[a], y[a], 'bo')
    if -1.75<x[a]<-1.5:
        plt.plot(x[a], y[a], 'mo')
    if -1.5<x[a]<-1.25:
        plt.plot(x[a], y[a], 'yo')
    if -1.25<x[a]<-1:
        plt.plot(x[a], y[a], 'co')
    
    plt.plot(x, y, lw=0.5) #plots rays 
    
    

    
plt.xlim(-2,2) #confine plot to a suitable region
plt.ylim(-2,2)



#circle outline
th = np.linspace(0,2*np.pi,100)
x1 = np.cos(th)
x2 = np.sin(th)
plt.plot(x1,x2,'k')
#circle fill with Monte carlo 
N = int(100000)
xc, yc  = 2*np.random.rand(2, N) - 1 
rad = np.sqrt(xc**2 + yc**2)
incirc = rad <= 1 #point in circle if below radius 1
plt.plot(xc[incirc==True], yc[incirc==True], "kx")









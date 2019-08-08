# Authors - Alex Reeves, Hannah Riley, Hong Qi

import numpy as np
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import odeint
from PIL import Image
import os
import scipy.misc as smp
from PIL import ImageFont
from PIL import ImageDraw
import warnings
warnings.filterwarnings("ignore")



#from IPython.display import Image
spin = 0
pixdensity = 5
view = np.pi/2 + 0.1



def raytracer(view, spin, pixdensity):
    #stepsize for RK4 and for ODEINT

    Rdisk = 20 # radius of the disk
    M = 1     # normalized for ease
    a = spin # angmom in theta direction
    #initial conditions spherical coords(at point of observer)
    r0 = 100*M
    theta0 = view#np.pi/2 + 0.1
    phi0 = 0 #np.pi/2
    pixelcoeff = 0.005 # similar to aperture and set as 0.025 or 0.05 when r0=100M. Scale accordingly - larger r0 smaller pixelcoeff
    h = 0.001
    tfinal = 1.3*r0
    N = int(tfinal/h)
    npts = 500
    tarray = np.arange(N+1)
    t = np.vstack(tarray)
    #figures
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    density = pixdensity # total number of pixels is density by density
    pixelspace = numpy.zeros((density**2,2))
    for i in numpy.arange(1,density+1):
        for j in numpy.arange(1,density+1):
            pixelspace[(i-1)*density+j-1]=[i,j]
    array = numpy.zeros([density, density, 3], dtype=numpy.uint8)

    for l in pixelspace: #np.arange(0,density): #multiple rays
        #print(l)
        # x and y, pixel space coords hmmmmm this could be reworked
        # Where from??! can just multiply by 5? NEED TO make sure range never over 2pi?
        xo = (l[0] - (density + 1.0) / 2) * pixelcoeff*r0/ (density - 1.0) #looping pixel coords hence ang and rad momentum
        yo = (l[1] - (density + 1.0) / 2) * pixelcoeff*r0 / (density - 1.0)

        # xo = (l[0] - 1 - (density/2))*pixelcoeff #looping pixel coords hence ang and rad momentum
        # yo = (l[1] - 1 - (density/2))*pixelcoeff
        # xo = (l[0] - (density + 1.0) / 2) * 2 * np.pi / (density - 1.0) #looping pixel coords hence ang and rad momentum
        # yo = (l[1] - (density + 1.0) / 2) * 2 *  np.pi / (density - 1.0)
        # xo, yo = -np.pi/2,-np.pi/2
        #print("xo, yo", xo, yo)

        #empty array for initial conditions
        y0 = np.zeros((1,6),float)

        #empty 2d array for variables
        y = np.zeros((N+1,6), float)

        #the variables used in the 5 ODEs
        #assigned to 2d vector y
        #NOTE: these are NOT primed
        t = y[:,0]
        r = y[:,1]
        theta = y[:,2]
        phi = y[:,3]
        pr = y[:,4]
        ptheta = y[:,5]


        #assign initial conditions to 2d array y0
        # (r0,theta0,phi0) define observer location
        y0[0,0] = t[0]
        y0[0,1] = r0
        y0[0,2] = theta0
        y0[0,3] = phi0

        #defining useful terms
        dell0 = r0**2 - 2*M*r0 + a**2
        sigma0 = r0**2 + a**2*np.cos(theta0)**2
        #later can define and add invsigma
        s = sigma0 - 2*r0


        #replace aperture by r0, fix the K constant term, fix bugs in phiprime and prprime definitions


        #NOTE: here below we get initial prprime0 and pthetaprime0
        #as we use the initial conditions for pr and ptheta0 from the 1st and 2nd odes
        #then we get prprime0 and pthetaprime0 by dividing by e
        #note he sets rprime0 (rdot0) as cos(x0)cos(y0)?
        #and thetaprime0 as sin(y0)?  pixel space conversions?
        rprime0 = np.cos(yo)*np.cos(xo)
        thetaprime0 = np.sin(yo)/r0
        #why is phiprime0 also defined by this
        phiprime0 = np.cos(yo) * np.sin(xo) / (r0 * np.sin(theta0))
        e = np.sqrt((s*(rprime0**2/dell0) + s*(thetaprime0**2) + dell0*(np.sin(theta0)**2)*phiprime0**2)) #energy
        #print(e)

        y0[0,4] = rprime0*(sigma0/dell0)/e      #from the 1st ODE, gives initial pr value
        y0[0,5] = thetaprime0*sigma0/e          #from 2nd ODE gives initial ptheta value

        #set first row in y vector as initial conditions
        y[0,:] = y0
        #L = ((sigma*delta*phidot0-2.0*a*r0*energy)*sin2/s1)/energy


        L = ((sigma0 * dell0 * phiprime0 - 2.0 * a * r0 * e) * (np.sin(theta0)**2)) / (s*e)
        K =  (thetaprime0*sigma0/e)**2 + (a*a) * (np.sin(theta0)**2) + (L * L) / (np.sin(theta0)**2)#carters constant
        #y0[0,4] * y0[0,4] was in place instead of the first term in K online
        #print("meh LK",L,K)
        #print("meow dell0, sigma0, s, phiprime0", dell0, sigma0, s, phiprime0)
        # we can reduce from 8 to 5 ODE's as t', pphi, pt'
        #can be ignored - dont affect trajectory
        #we shall integrate 5 ODES with RK4
        rh = 1+np.sqrt(1-a*a)# event horizon

        # def rprime(r, pr, theta):
        #     dell = r**2 - 2*M*r + a**2
        #     sigma = r**2 + a**2*(np.cos(theta)**2)
        #     if r<rh: #doesnt enter event horizon
        #         return np.inf # this need to be fixed
        #     return -(dell/sigma)*pr

        # def thetaprime(r,theta,ptheta):
        #     sigma = r**2 + a**2*(np.cos(theta)**2)
        #     return -(1/sigma)*ptheta

        # def phiprime(r, theta):
        #     dell = r**2 - 2*M*r + a**2
        #     sigma = r**2 + a**2*(np.cos(theta)**2)
        #     return -(2*a*r + (sigma - 2*r)*L/np.sin(theta)**2 ) / (dell*sigma)

        # def prprime(r, pr, theta):
        #     dell = r**2 - 2*M*r + a**2
        #     sigma = r**2 + a**2*(np.cos(theta)**2)
        #     return -(((r-1)*(-K) + 2*r*(r**2 + a**2) - 2*a*L)/(dell*sigma) - (2*pr**2*(r-1)/sigma))

        # def pthetaprime(r, theta):
        #     sigma = r**2 + a**2*(np.cos(theta)**2)
        #     return -(np.sin(theta)*np.cos(theta)*(L**2/np.sin(theta)**2 - a**2))/sigma #on his online write up he has L**2/(a**2) but this seems not to work
        a2 = a*a
        # Evolving the orbit with scipy function odeint
        def solver(z,tao):
            r = z[0]
            r2=r*r
            theta = z[1]
            if theta < 1e-8: #stops theta going to completely to zero
                theta = 1e-8
            costh=np.cos(theta)
            cos2=costh*costh
            phi = z[2]
            pr = z[3]
            ptheta = z[4]
            sigma = r2 + a2*cos2 #solver has been optimized by removing prime functions for expressions
            dell = r2 - 2*M*r + a2

            drdtao = -(dell/sigma)*pr
            if r<(rh+0.1): drdtao=-np.inf#doesnt enter event horizon


            dthetadtao = -(1/sigma)*ptheta

            dphidtao = -(2*a*r + (sigma - 2*r)*L/np.sin(theta)**2 ) / (dell*sigma)

            dprdtao = -(((r-1)*(-K) + 2*r*(r**2 + a**2) - 2*a*L)/(dell*sigma) - (2*pr**2*(r-1)/sigma))

            dpthetadtao = -(np.sin(theta)*np.cos(theta)*(L**2/np.sin(theta)**2 - a**2))/sigma
            dzdtao = [drdtao, dthetadtao, dphidtao, dprdtao, dpthetadtao]
            return dzdtao
        # initial condition
        z0 = [r0, theta0, phi0, y0[0,4], y0[0,5]]
        #print(z0)
        # proper time points
        # tao = np.linspace(0,100000)*h
        # can define steps in tau
        tao = np.linspace(0,tfinal,npts)

        # solve ODE
        z = odeint(solver, z0, tao) #mxstep = 5000000)
        r = z[:,0]
        theta = z[:,1]
        phi = z[:,2]

        # Evolving trajectory with RK4, here using "continue" to comment out so as to use odeint method
        for i in range(N):
            continue
            t[i+1] = t[i] + h #missing this independent variable

            k1rprime = h*rprime(r[i], pr[i], theta[i])
            k1thetaprime = h*thetaprime(r[i], theta[i], ptheta[i])
            k1phiprime = h*phiprime(r[i], theta[i])
            k1prprime = h*prprime(r[i], pr[i], theta[i])
            k1pthetaprime = h*pthetaprime(r[i], theta[i])

            k2rprime = h*rprime(r[i] + 1/2*k1rprime, pr[i] + 1/2*k1prprime, theta[i] + 1/2*k1thetaprime)
            k2thetaprime = h*thetaprime(r[i]+1/2*k1rprime, theta[i] + 1/2*k1thetaprime, ptheta[i]+1/2*k1pthetaprime)
            k2phiprime = h*phiprime(r[i]+1/2*k1rprime, theta[i]+1/2*k1thetaprime)
            k2prprime = h*prprime(r[i]+1/2*k1rprime, pr[i]+1/2*k1prprime, theta[i]+1/2*k1thetaprime)
            k2pthetaprime = h*pthetaprime(r[i]+1/2*k1rprime, theta[i]+1/2*k1thetaprime)

            k3rprime = h*rprime(r[i] + 1/2*k2rprime, pr[i] + 1/2*k2prprime, theta[i] + 1/2*k2thetaprime)
            k3thetaprime = h*thetaprime(r[i] + 1/2*k2rprime, theta[i] + 1/2*k2thetaprime, ptheta[i] + 1/2*k2pthetaprime)
            k3phiprime = h*phiprime(r[i] + 1/2*k2rprime, theta[i] + 1/2*k2thetaprime)
            k3prprime = h*prprime(r[i] + 1/2*k2rprime, pr[i] + 1/2*k2prprime, theta[i] + 1/2*k2thetaprime)
            k3pthetaprime = h*pthetaprime(r[i] + 1/2*k2rprime, theta[i] + 1/2*k2thetaprime)

            k4rprime = h*rprime(r[i] + k3rprime, pr[i] + k3prprime, theta[i] + k3thetaprime)
            k4thetaprime = h*thetaprime(r[i] + k3rprime, theta[i] + k3thetaprime, ptheta[i] + k3pthetaprime)
            k4phiprime = h*phiprime(r[i] + k3rprime, theta[i] + k3thetaprime)
            k4prprime = h*prprime(r[i] + k3rprime, pr[i] + k3prprime, theta[i] + k3thetaprime)
            k4pthetaprime = h*pthetaprime(r[i] + k3rprime, theta[i] + k3thetaprime)

            r[i+1] = r[i]+1/6*(k1rprime + 2*k2rprime + 2*k3rprime + k4rprime)
            theta[i+1] = theta[i]+1/6*(k1thetaprime + 2*k2thetaprime + 2*k3thetaprime + k4thetaprime)
            phi[i+1] = phi[i]+1/6*(k1phiprime + 2*k2phiprime + 2*k3phiprime + k4phiprime)
            pr[i+1] = pr[i]+1/6*(k1prprime + 2*k2prprime + 2*k3prprime + k4prprime)
            ptheta[i+1] = ptheta[i]+1/6*(k1pthetaprime + 2*k2pthetaprime + 2*k3pthetaprime + k4pthetaprime)


        # convert back to cartesian
        xc = np.sqrt(r**2)*np.sin(phi)*np.sin(theta)
        yc = np.sqrt(r**2)*np.cos(phi)*np.sin(theta)
        zc = np.sqrt(r**2)*np.cos(theta)

        ax.scatter3D(xc,yc,zc, '.', label=[xo,yo], s = 1)
        ax.scatter3D(xc[0],yc[0],zc[0], s=212) #observer location
        ax.view_init(elev=10., azim=30) # azim is the 3D plot view angle, its range is 0 to 360
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        sidex = 10
        sidey = 10
        sidez = 10
        ax.set_xlim3d(-sidex,sidex)
        ax.set_ylim3d(-sidey,sidey)
        ax.set_zlim3d(-sidez,sidez)
        ax.legend(loc='best')
        minr = min(r)
        #print(minr)
        xindex = np.int(l[0] - 1)
        yindex = np.int(l[1] - 1)
        #print(xindex, yindex)
        # assign colors to pixels. Below is rough, need to be fixed. Need to know
        #if the back-traced rays encounter the disk.
        tolerance = 0.1 #fudge factor as above screws up near r = 2!
        #looking at r vals for rays can see that r value minimum is anywhere from
        #2.01 to 2.09 so have to set this tolerance!
        # if (minr < rh + tolerance or minr > Rdisk):
        #     array[xindex, yindex] = [0, 0, 0] #black
        # else:
        #     array[xindex, yindex] = [255, 255, 255]   # white
        array[xindex, yindex] = [0, 0, 0]





        #COLLISION DETECTION- DODGY MIGHT NEED CHANGES, check with simple case...

        for i in range(1,len(r)-5):

            # while rh + 0.1 < r[i] < Rdisk:
            #     print(theta[i])
            if (theta[i] - np.pi/2)*(theta[i+1]-np.pi/2) < 0 or (theta[i] + np.pi/2)*(theta[i+1] + np.pi/2) < 0:
                if rh + tolerance < r[i] < Rdisk:
                    array[xindex, yindex] = [255, 255, 255]
                    #print('yes')


        #code for plotting r and theta vals for one ray- CAN SEE THAT THETA GOES NEGATIVE FOR SOME RAYS
        # if (xo,yo) == (0,0.125):
        #     ax1 = fig.add_subplot(111)
        #     ax1.plot(tao,theta)














    #PLOTTING
    #Making black hole sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = rh * np.outer(np.cos(u), np.sin(v))
    y = rh * np.outer(np.sin(u), np.sin(v))
    z = rh * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, rstride=16, cstride=16, color='k', linewidth=3, alpha=0.5)
    iters = np.arange(0,100)
    iter = np.random.choice(iters,1)
    # plt.savefig("kerr%d.png" % iter)
    # plt.show()



    reduced = np.empty([pixdensity,pixdensity]) #shape of pixdensity x pixdensity
    #counter=0
    #counter2=0
    for i in range(len(array)): #reshapes from e.g. 10,10,3 to 10,10 (ie sums the 3)
        #counter2+=1
        for j in range(len(array[i,:])):
            #counter+=1
            reduced[i,j]=np.sum(array[i,j]) #assigns to new array
    binreduced = (reduced!=0).astype(int) #makes values 0 or 1
    binaryflat = binreduced.flatten() #1D

    #Creating image
    #img = smp.toimage(array)       # Create a PIL image
    #img.show()                     # View in default viewer
    #img.save('doingcolls.png')
    return binaryflat #returns 1d binary array length of pixdensity^2

# array = raytracer(spin, pixdensity)
# print((array))



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


#stepsize for RK4
h = 0.001
tfinal = 1
N = int(tfinal/h)
tarray = np.arange(N+1)
t = np.vstack(tarray)



#defining variables, hardcoded
M = 1     # normalized for ease
a = 1 #angmom in theta direction
aperture = 0.5

#figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for l in range(3): #multiple rays
    # x and y, pixel space coords
    xo = l*np.pi/2 #looping pixel coords hence ang and rad momentum
    yo = l*np.pi/2

    #empty array for initial conditions
    y0 = np.zeros((1,6),float)

    #empty 2d array for variables
    y = np.zeros((N+1,6), float)

    #initial conditions spherical coords(at point of observer)
    r0 = 6*M
    theta0 = np.pi/2
    phi0 = 0

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
    sigma0 = r0**2 + a**2*(np.cos(theta0)**2)
    #later can define and add invsigma
    s = sigma0 - 2*r0


    #NOTE: here below we get initial prprime0 and pthetaprime0
    #as we use the initial conditions for pr and ptheta0 from the 1st and 2nd odes
    #then we get prprime0 and pthetaprime0 by dividing by e
    #note he sets rprime0 (rdot0) as cos(x0)cos(y0)?
    #and thetaprime0 as sin(y0)?  pixel space conversions?
    rprime0 = np.cos(yo)*np.cos(xo)
    thetaprime0 = np.sin(yo)/aperture
    #why is phiprime0 also defined by this
    phiprime0 = np.cos(yo) * np.sin(xo) / (aperture * np.sin(theta0))
    e = np.sqrt((s*(rprime0**2/dell0) + s*(thetaprime0**2) + dell0*(np.sin(theta0)**2)*phiprime0**2)) #energy


    y0[0,4] = (rprime0*(sigma0/dell0))/e      #from the 1st ODE, gives initial pr value
    y0[0,5] = (thetaprime0*sigma0)/e          #from 2nd ODE gives initial ptheta value

    #set first row in y vecror as initial conditions
    y[0,:] = y0


    L = ((sigma0 * dell0 * phiprime0 - 2.0 * a * r0 * e) * (np.sin(theta0)**2)) / (s*e)
    K =  (thetaprime0*sigma0)**2 + (a*a) * (np.sin(theta0)**2) + (L * L) / (np.sin(theta0)**2)#carters constant
    #y0[0,4] * y0[0,4] was in place instead of the first term in K online

    # we can reduce from 8 to 5 ODE's as t', pphi, pt'
    #can be ignored - dont affect trajectory
    #we shall integrate 5 ODES with RK4

    #defining ODE's

    def rprime(r, pr, theta):
        dell = r**2 - 2*M*r + a**2
        sigma = r**2 + a**2*(np.cos(theta)**2)

        if r<2: #doesnt enter event horizon
            return np.inf
        return -(dell/sigma)*pr

    def thetaprime(r,theta,ptheta):
        sigma = r**2 + a**2*(np.cos(theta)**2)
        return -(1/sigma)*ptheta

    def phiprime(r, theta):
        dell = r**2 - 2*M*r + a**2
        sigma = r**2 + a**2*(np.cos(theta)**2)
        return -(2*a*r + (((sigma - 2*r)*L)/(np.sin(theta)**2))/(dell*sigma))

    def prprime(r, pr, theta):
        dell = r**2 - 2*M*r + a**2
        sigma = r**2 + a**2*(np.cos(theta)**2)
        return -(((r-1)*(-K) + 2*r*(r**2 + a**2) - 2*a*L)/(dell*sigma)) - ((2*(pr**2)*(r-1))/sigma)

    def pthetaprime(r, theta):
        sigma = r**2 + a**2*(np.cos(theta)**2)
        return -(np.sin(theta)*np.cos(theta)*(L**2/((np.sin(theta))**2) - a**2))/sigma #on his online write up he has L**2/(a**2) but this seems not to work


    for i in range(N):
        #t[i+1] = t[i] + h missing this independent variable

        k1rprime = rprime(r[i], pr[i], theta[i])
        k1thetaprime = thetaprime(r[i], theta[i], ptheta[i])
        k1phiprime = phiprime(r[i], theta[i])
        k1prprime = prprime(r[i], pr[i], theta[i])
        k1pthetaprime = pthetaprime(r[i], theta[i])

        k2rprime = rprime(r[i] + h/2*k1rprime, pr[i] + h/2*k1prprime, theta[i] + h/2*k1thetaprime)
        k2thetaprime = thetaprime(r[i]+h/2*k1rprime, theta[i] + h/2*k1thetaprime, ptheta[i]+h/2*k1pthetaprime)
        k2phiprime = phiprime(r[i]+h/2*k1rprime, theta[i]+h/2*k1thetaprime)
        k2prprime = prprime(r[i]+h/2*k1rprime, pr[i]+h/2*k1prprime, theta[i]+h/2*k1thetaprime)
        k2pthetaprime = pthetaprime(r[i]+h/2*k1rprime, theta[i]+h/2*k1thetaprime)

        k3rprime = rprime(r[i] + h/2*k2rprime, pr[i] + h/2*k2prprime, theta[i] + h/2*k2thetaprime)
        k3thetaprime = thetaprime(r[i] + h/2*k2rprime, theta[i] + h/2*k2thetaprime, ptheta[i] + h/2*k2pthetaprime)
        k3phiprime = phiprime(r[i] + h/2*k2rprime, theta[i] + h/2*k2thetaprime)
        k3prprime = prprime(r[i] + h/2*k2rprime, pr[i] + h/2*k2prprime, theta[i] + h/2*k2thetaprime)
        k3pthetaprime = pthetaprime(r[i] + h/2*k2rprime, theta[i] + h/2*k2thetaprime)

        k4rprime = rprime(r[i] + h*k3rprime, pr[i] + h*k3prprime, theta[i] + h*k3thetaprime)
        k4thetaprime = thetaprime(r[i] + h*k3rprime, theta[i] + h*k3thetaprime, ptheta[i] + h*k3pthetaprime)
        k4phiprime = phiprime(r[i] + h*k3rprime, theta[i] + h*k3thetaprime)
        k4prprime = prprime(r[i] + h*k3rprime, pr[i] + h*k3prprime, theta[i] + h*k3thetaprime)
        k4pthetaprime = pthetaprime(r[i] + h*k3rprime, theta[i] + h*k3thetaprime)

        r[i+1] = r[i]+h/6*(k1rprime + 2*k2rprime + 2*k3rprime + k4rprime)
        theta[i+1] = theta[i]+h/6*(k1thetaprime + 2*k2thetaprime + 2*k3thetaprime + k4thetaprime)
        phi[i+1] = phi[i]+h/6*(k1phiprime + 2*k2phiprime + 2*k3phiprime + k4phiprime)
        pr[i+1] = pr[i]+h/6*(k1prprime + 2*k2prprime + 2*k3prprime + k4prprime)
        ptheta[i+1] = ptheta[i]+h/6*(k1pthetaprime + 2*k2pthetaprime + 2*k3pthetaprime + k4pthetaprime)


    # convert back to cartesian
    xc = np.sqrt(a**2+r**2)*np.cos(phi)*np.sin(theta)
    yc = np.sqrt(a**2+r**2)*np.sin(phi)*np.sin(theta)
    zc = np.sqrt(a**2+r**2)*np.cos(theta)

    ax.scatter3D(xc,yc,zc, label=[xo,yo], s = 0.5)
    ax.set_xlim3d(-10,10)
    ax.set_ylim3d(-10,10)
    ax.set_zlim3d(-10,10)
    ax.legend(loc='best')


print("Initial conditions are", y[0,:])






#Making black hole sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = 2 * np.outer(np.cos(u), np.sin(v))
y = 2 * np.outer(np.sin(u), np.sin(v))
z = 2 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='k', linewidth=0, alpha=0.3)
plt.show()

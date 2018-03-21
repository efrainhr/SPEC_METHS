#!/usr/bin/python

import os
import numpy as np
import matplotlib.pylab as pl

pi = np.pi

from scipy.linalg import toeplitz 

def vim(fn='spectral_mets.py'): os.system('vim '+fn)

def p1():
    from scipy.sparse import diags

    Nv=2**np.arange(3,17)

    for n in Nv:
        h=2*pi/n
        x=-pi+h*np.arange(n)
        u=np.exp(np.sin(x))
        uprime=np.cos(x)*u #du/dx

        d1=np.array([2/3.])
        d2=1/12.+np.zeros(2)
        d3=1/12.+np.zeros(n-2)
        d4=2/3.+np.zeros(n-1)
        D=diags([d1,-d2,d3,-d4,d4,-d3,d2,-d1],\
                [-n+1,-n+2,-2,-1,1,2,n-2,n-1])
        #print np.round(D.toarray(),decimals=2)

        D=D/h

        error=np.linalg.norm(D*u-uprime)
        pl.loglog(n,error,'o',c='#1f77b4')

        d1=np.array([0.5])
        d2=0.5+np.zeros(n-1)
        D=diags([d1,-d2,d2,-d1],[-n+1,-1,1,n-1])

        D=D/h

        error=np.linalg.norm(D*u-uprime)
        pl.loglog(n,error,'o',c='#2ca02c')

        #d=[np.zeros(n-i-1)+1./(i+1) for i in range(n-1)]
        #diag=[d1 if i%2 else -d1 for i, d1 in enumerate(reversed(d))]+\
        #     [-d1 if i%2 else d1 for i, d1 in enumerate(d)]
        #loc=[-i for i in range(n-1,0,-1)]+range(1,n)
        #D=diags(diag,loc).toarray()
        #if n==16: print np.round(D.toarray(), decimals=3)

        #D=D/h

        #error=np.linalg.norm(np.dot(D,u)-uprime)
        #pl.loglog(n,error,'o',c='#bcbd22')

    pl.semilogy(Nv,1./Nv**4,'--',c='#1f77b4')
    pl.semilogy(Nv,1./Nv**2,'--',c='#2ca02c')
    pl.title('Convergence of  2- & 4-th order finite differences')
    pl.xlabel('N')
    pl.grid(ls='--',which='both')
    pl.text(20,5e-8,r'N$^{-4}$',fontsize=14)
    pl.text(2000,1e-6,r'N$^{-2}$',fontsize=14)
    pl.ylabel('error')
    pl.show()

def p2():
    from scipy.linalg import toeplitz

    Nv=2**np.arange(3,13)

    for n in np.arange(2,102,2):
        h=2*pi/n
        x=-pi+h*np.arange(n)
        u=np.exp(np.sin(x))
        uprime=np.cos(x)*u #du/dx

        col=np.append(0,0.5*(-1)**np.arange(1,n)/np.tan(np.arange(1,n)*h/2.))

        D=toeplitz(col,col[[0]+range(n-1,0,-1)])

        error=np.linalg.norm(np.dot(D,u)-uprime,ord=np.inf)
        pl.loglog(n,error,'o',c='#1f77b4')

    pl.title('Convergence of spectral differentiation')
    pl.xlabel('N')
    pl.grid(ls='--',which='both')
    pl.ylabel('error')
    pl.show()

def p3():
    h=1.; xmax=10

    x=np.arange(-xmax,xmax+1,1)
    xx=np.arange(-xmax-h/20,xmax+(1+h)/20,h/10)

    fig, axes = pl.subplots(nrows=3)

    b=np.zeros(len(x))
    v={2: np.maximum(0,1-abs(x)/3.)}
    v[0]=b.copy(); v[0][x==0]=1
    v[1]=b.copy(); v[1][abs(x)<=3]=1

    for n in range(3):
        p=np.zeros(len(xx))
        for i in range(len(x)):
            p+=v[n][i]*np.sin(pi*(xx-x[i])/h)/(pi*(xx-x[i])/h)
        axes[n].plot(xx,p)
        axes[n].plot(x,v[n],'o',c='#9467bd')
        axes[n].set_xticks([])

    axes[2].set_xticks([xt for xt in x if not xt%2])
    pl.show()

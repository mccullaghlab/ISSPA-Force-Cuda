import numpy as np

qm=1.00
qp=1.00
d0=-0.21

rC=abs(d0)
rH=abs(1.1+d0)
rCl=np.sqrt(1.758**2+d0**2+2.*d0*1.758*np.cos(1.87937134))
if abs(d0) < 1.e-8:
    cC=1.
else:
    cC=d0/rC
cH=(1.1+d0)/rH
if abs(d0) < 1.e-8:
    cCl=np.cos(1.87937134)
else:
    cCl=(d0**2+rCl**2-1.758**2)/2./d0/rCl
qH=0.2659
qCl=0.0396*3.
qC=-qH-qCl

dipo=qH*rH*cH+qCl*rCl*cCl+qC*rC*cC
quad=qH*rH**2*(1.5*cH**2-0.5)+qCl*rCl**2*(1.5*cCl**2-0.5)+qC*rC**2*(1.5*cC**2-0.5)
octu=qH*rH**3*(2.5*cH**2-1.5)*cH+qCl*rCl**3*(2.5*cCl**2-1.5)*cCl+qC*rC**3*(2.5*cC**2-1.5)*cC

eps=2.35
rho=0.0074

pLJdat=np.loadtxt("LJ.qp{:4.2f}.dn0.21.gr1".format(qp))
mLJdat=np.loadtxt("LJ.qm{:4.2f}.dn0.21.gr1".format(qm))
pCdat=np.loadtxt("LJ.qp{:4.2f}.dn0.21.gr1".format(qp))
mCdat=np.loadtxt("LJ.qm{:4.2f}.dn0.21.gr1".format(qm))

r0=np.append(pLJdat[:,0],[0.,0.])
gLJp=np.append(pLJdat[:,1],[1.,1.])
gLJp=np.log(gLJp,out=-1.E15*np.ones_like(gLJp),where=gLJp!=0.)
gCp=np.append(pCdat[:,1],[1.,1.])
gCp=np.log(gCp,out=-1.E15*np.ones_like(gCp),where=gCp!=0.)
fp=np.append(pLJdat[:,2],[0.,0.])
fp[np.where(np.isnan(fp))]=0.
pp=np.append(pCdat[:,4],[0.,0.])
pp[np.where(np.isnan(pp))]=0.

gLJm=np.append(mLJdat[:,1],[1.,1.])
gLJm=np.log(gLJm,out=-1.E15*np.ones_like(gLJm),where=gLJm!=0.)
gCm=np.append(mCdat[:,1],[1.,1.])
gCm=np.log(gCm,out=-1.E15*np.ones_like(gCm),where=gCm!=0.)
fm=np.append(mLJdat[:,2],[0.,0.])
fm[np.where(np.isnan(fm))]=0.
pm=np.append(mCdat[:,4],[0.,0.])
pm[np.where(np.isnan(pm))]=0.

Ep=pp/(1.-pp**2)*(3.-(6.*pp**2+pp**4-2.*pp**6)/5.)
Em=pm/(1.-pm**2)*(3.-(6.*pm**2+pm**4-2.*pm**6)/5.)

xx,yy=np.meshgrid(np.arange(-99.95,100,0.1),np.arange(0.05,25.,0.1))

fpC=np.zeros(500)
fmC=np.zeros(500)
fpLJ=np.zeros(500)
fmLJ=np.zeros(500)
upC=np.zeros(500)
umC=np.zeros(500)
upLJ=np.zeros(500)
umLJ=np.zeros(500)
for iR,R in enumerate(np.arange(0.1,50.1,0.1)):
#for iR,R in enumerate(np.arange(0.1,15.1,0.1)):
#for R in np.arange(15.,15.25,0.25):
    hR=R/2.
   
    rp=np.sqrt((xx+hR)**2+yy**2)
    drp=rp/0.1-0.5
    irp=np.ndarray.astype(np.floor(drp),np.int)
    drp-=irp
    irp[np.where(irp>248)]=250
    irp[np.where(irp<0)]=0
    
    rm=np.sqrt((xx-hR)**2+yy**2)
    drm=rm/0.1-0.5
    irm=np.ndarray.astype(np.floor(drm),np.int)
    drm-=irm
    irm[np.where(irm>248)]=250
    irm[np.where(irm<0)]=0
    
    gLJ=np.exp(gLJp[irp]+drp*(gLJp[irp+1]-gLJp[irp])+gLJm[irm]+drm*(gLJm[irm+1]-gLJm[irm]))
    gC=np.exp(gCp[irp]+drp*(gCp[irp+1]-gCp[irp])+gCm[irm]+drm*(gCm[irm+1]-gCm[irm]))
    
    Ex=(xx+hR)*(Ep[irp]+drp*(Ep[irp+1]-Ep[irp]))/rp+(xx-hR)*(Em[irm]+drm*(Em[irm+1]-Em[irm]))/rm
    Ey=    yy *(Ep[irp]+drp*(Ep[irp+1]-Ep[irp]))/rp+    yy *(Em[irm]+drm*(Em[irm+1]-Em[irm]))/rm
    
    Ex+=np.divide(-(xx+hR)*(1.-1./eps)*3./4./np.pi/rho/dipo,rp**3,out=np.zeros_like(Ex),where=(irp==250) & (irm!=250))
    Ex+=np.divide( (xx-hR)*(1.-1./eps)*3./4./np.pi/rho/dipo,rm**3,out=np.zeros_like(Ex),where=(irm==250) & (irp!=250))
    Ey+=np.divide(-yy*(1.-1./eps)*3./4./np.pi/rho/dipo,rp**3,out=np.zeros_like(Ex),where=(irp==250) & (irm!=250))
    Ey+=np.divide( yy*(1.-1./eps)*3./4./np.pi/rho/dipo,rm**3,out=np.zeros_like(Ex),where=(irm==250) & (irp!=250))

    Etot=np.sqrt(Ex**2+Ey**2)
    px=np.divide(Ex,Etot,out=np.zeros_like(Etot),where=Etot!=0.)
    py=np.divide(Ey,Etot,out=np.zeros_like(Etot),where=Etot!=0.)

    Rzp=(px*(xx+hR)+py*yy)/rp
    Rzm=(px*(xx-hR)+py*yy)/rm
    
    cothE=np.divide(1.,np.tanh(Etot),out=np.zeros_like(Etot),where=Etot!=0.)
    Einv=np.divide(1.,Etot,out=np.zeros_like(Etot),where=Etot!=0.)

    c1=cothE-Einv
    c2=1.-2.*Einv*c1
    c3=cothE-3.*Einv*c2
    
    #for i in range(len(xx[:,0])):
    #    for j in range(len(xx[0])):
    #        print(xx[i,j],yy[i,j],g[i,j],px[i,j],py[i,j])
    
    fpx=gLJ*((fp[irp]+drp*(fp[irp+1]-fp[irp]))*(xx+hR)/rp)*yy
    fmx=gLJ*((fm[irm]+drm*(fm[irm+1]-fm[irm]))*(xx-hR)/rm)*yy
    
    fpLJ[iR]= 2.*np.pi*rho*0.1**2*np.sum(fpx)
    fmLJ[iR]=-2.*np.pi*rho*0.1**2*np.sum(fmx)


    #dipole
    fpx= dipo*c1/rp**3*(3.*Rzp*(xx+hR)/rp-px)
    fmx=-dipo*c1/rm**3*(3.*Rzm*(xx-hR)/rm-px)
    #quadrupole
    fpx+= quad*(1.5*c2-0.5)/rp**4*((7.5*Rzp**2-1.5)*(xx+hR)/rp-3.*Rzp*px)
    fmx+=-quad*(1.5*c2-0.5)/rm**4*((7.5*Rzm**2-1.5)*(xx-hR)/rm-3.*Rzm*px)
    #octupole
    fpx+= octu*(2.5*c3-1.5*c1)/rp**5*((17.5*Rzp**3-7.5*Rzp)*(xx+hR)/rp-(7.5*Rzp**2-1.5)*px)
    fmx+=-octu*(2.5*c3-1.5*c1)/rm**5*((17.5*Rzm**3-7.5*Rzm)*(xx-hR)/rm-(7.5*Rzm**2-1.5)*px)
    #scale
    fpx*=gC*yy
    fmx*=gC*yy
    
    fpC[iR]= 2.*332.*qp*np.pi*rho*0.1**2*np.sum(fpx)
    fmC[iR]=-2.*332.*qm*np.pi*rho*0.1**2*np.sum(fmx)

hdx=0.05
#hdx=0.05
for i in range(499):
    upC[i+1]=upC[i]-hdx*(fpC[i]+fpC[i+1])
    umC[i+1]=umC[i]-hdx*(fmC[i]+fmC[i+1])
    upLJ[i+1]=upLJ[i]-hdx*(fpLJ[i]+fpLJ[i+1])
    umLJ[i+1]=umLJ[i]-hdx*(fmLJ[i]+fmLJ[i+1])

upC-=upC[499]
umC-=umC[499]
upLJ-=upLJ[499]
umLJ-=umLJ[499]

fout=open("LJ2.qp{:4.2f}.qm{:4.2f}.IS-SPA.dn0.21.dat".format(qp,qm),"w")
for i in range(500):
    fout.write("{:6.3f} {:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e}\n".format(0.1*(i+1),fpLJ[i],fmLJ[i],fpC[i],fmC[i],upLJ[i],umLJ[i],upC[i],umC[i]))

fout.close()

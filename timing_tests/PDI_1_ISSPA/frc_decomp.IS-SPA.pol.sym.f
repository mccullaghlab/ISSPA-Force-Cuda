CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C  Things to note:
C     - H mass = 12
C     - no interaction cut-off
C     - thermostat:  Andersen thermostat
C     - solvent sampling: uniform
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      PROGRAM MD_CHX
        include 'omp_lib.h'
        integer nmax,hmax
        parameter(nmax=2000,hmax=120)
C paramtop values
        integer natom,nbonh,nbona,ntheth,ntheta
        integer nphih,nphia,nnb,nres,ntyp,nmol
        double precision charge(nmax),mass(nmax)
        integer ityp(nmax),nbparm(nmax),nrpnt(nmax),nmpnt(nmax)
        double precision kbnd(nmax),rbnd(nmax),kang(nmax),tang(nmax)
        double precision kdih(nmax),pdih(nmax),ndih(nmax)
        double precision scee(nmax),scnb(nmax)
        double precision alj(nmax),blj(nmax)
        integer bhlist(3,nmax),balist(3,nmax)
        integer ahlist(4,nmax),aalist(4,nmax)
        integer dhlist(5,nmax),dalist(5,nmax)
        integer excl(nmax),nexc(nmax)
C solvent values
        integer npts,ncpu,nstyp
        integer styp(nmax),imin(nmax)
        double precision gr(hmax,nmax),Emf(hmax,nmax)
        double precision fLJ(hmax,nmax),vtot(nmax)
        double precision e0,p0,rmax,eps
C random
        integer ma(55),inext,inextp
C
        double precision x(3,nmax),v(3,nmax),f(3,nmax,4)
        double precision lbox(3),hlbox(3),dt,hdt,T,ran3,try,pnu
C
        character*64 rst,frc,xyz,vel,ene,com
        character*64 frst,prmtop
C
        integer i,j,igo,jgo,kgo,tid,ip
        double precision r2,rcom(3)
C
        common /param/ charge,mass,kbnd,rbnd,kang,tang,kdih,pdih,ndih,
     &       scee,scnb,alj,blj,ityp,
     &       nbparm,nrpnt,nmpnt,bhlist,balist,ahlist,aalist,dhlist,
     &       dalist,excl,nexc,natom,ntyp,nbonh,nbona,ntheth,
     &       ntheta,nphih,nphia,nnb,nmol,nres
C
        common /solvent/ gr,Emf,fLJ,vtot,e0,p0,rmax,eps,
     &       imin,styp,npts,ncpu,nstyp
C
        common /random/ ma,inext,inextp
C$OMP threadprivate(/random/)
C
        call omp_set_dynamic(.false.)
        ncpu=6
        dt=0.002d0
        T=298.d0
C
        dt=dt*20.455d0
        hdt=dt*0.5d0
        T=T*0.00198717d0
        pnu=1.d0/30.d0
C  Read in the input file
        call getinput(prmtop,frst)
C  Get random number seed
        open(20,FILE='ran3.dat',STATUS='old')
        read(20,*) inext,inextp
        do i=1,55
          read(20,*) ma(i)
        enddo
        close(20)
C  Put different seed on each thread
C$OMP parallel num_threads(ncpu) default(none)
C$OMP& private(tid)
C$OMP& copyin(ma,inext,inextp)
        tid=omp_get_thread_num()
        inext=inext+tid
        if (inext.gt.55) inext=inext-55
        inextp=inextp+tid
        if (inextp.gt.55) inextp=inextp-55
C$OMP end parallel
C  Get the respective parameters for atom types
        call setparam(prmtop)
        call makemol()
C  Get the solvent effect parameters
        call setsolvent(charge)
C  Get the positions, velocities, and atom types
        call getx(x,v,lbox,hlbox,natom,frst,T,mass)
        call wrap(x,lbox,nmol,nmpnt,ncpu)
C  Get new forces
        call getfrc(x,lbox,hlbox,f)
        do i=1,natom
C          write(6,*) x(1,i),x(2,i),x(3,i),f(1,i,1),f(2,i,1),f(3,i,1)
C     &        ,f(1,i,2),f(2,i,2),f(3,i,2),f(1,i,3),f(2,i,3),f(3,i,3)
C     &        ,f(1,i,4),f(2,i,4),f(3,i,4)
          write(6,*) x(1,i),x(2,i),x(3,i),f(1,i,4),f(2,i,4),f(3,i,4)
        enddo
C  Put random number seed
        open(20,FILE='ran3.dat',STATUS='unknown')
        write(20,*) inext,inextp
        do i=1,55
          write(20,*) ma(i)
        enddo
        close(20)
C
      stop
      end
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      double precision function ran3()
        integer mbig,mseed,mz,mbig1,inext,inextp,mj
        double precision fac
        parameter(mbig=1000000000,mseed=161803398,mz=0)
        parameter(mbig1=mbig-1,fac=1.d0/mbig)
        integer ma(55)
C
        common /random/ ma,inext,inextp
C$OMP threadprivate(/random/)
C
        inext=inext+1
        if (inext.eq.56) inext=1
        inextp=inextp+1
        if (inextp.eq.56) inextp=1
        mj=ma(inext)-ma(inextp)
        if (mj.le.mz)mj=mj+mbig1
        ma(inext)=mj
        ran3=mj*fac
C
      return
      end
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      subroutine getx(x,v,lbox,hlbox,natom,frst,T,mass)
        integer nmax
        parameter(nmax=2000)
        double precision x(3,nmax),v(3,nmax)
        double precision lbox(3),hlbox(3),T
        double precision mass(nmax)
        integer i,ip,natom
        character*64 frst
C
        open(20,FILE=frst,STATUS='old')
        read(20,*)
        read(20,*)
        i=1
        ip=2
        do while (i.le.natom)
          read(20,999) x(1,i),x(2,i),x(3,i),x(1,ip),x(2,ip),x(3,ip)
          i=i+2
          ip=ip+2
        enddo
C        i=1
C        ip=2
C        do while (i.le.natom)
C          read(20,999) v(1,i),v(2,i),v(3,i),v(1,ip),v(2,ip),v(3,ip)
C          i=i+2
C          ip=ip+2
C        enddo
        read(20,998) lbox(1),lbox(2),lbox(3)
999     format(6(f12.7))
998     format(3(f12.7))
        close(20)
        do i=1,3
          hlbox(i)=lbox(i)*0.5d0
        enddo
C
      return
      end
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      subroutine wrap(x,lbox,nres,nrpnt,ncpu)
        integer nmax
        parameter(nmax=2000)
        double precision x(3,nmax)
        double precision lbox(3),dx
        integer nrpnt(nmax)
        integer i,j,k,nres,ndx,jmin,jmax,ncpu
C
C$OMP parallel do schedule(dynamic) num_threads(ncpu) default(none)
C$OMP& private(i,j,k,jmin,jmax,dx,ndx)
C$OMP& shared(nres,nrpnt,lbox,x)
        do i=1,nres
          jmin=nrpnt(i)
          do k=1,3
            dx=x(k,jmin)
            if (dx.gt.lbox(k)) then
              ndx=int(dx/lbox(k))
              jmax=nrpnt(i+1)-1
              do j=jmin,jmax
                x(k,j)=x(k,j)-dble(ndx)*lbox(k)
              enddo
            endif
            if (dx.lt.0.d0) then
              ndx=int(-dx/lbox(k)+1)
              jmax=nrpnt(i+1)-1
              do j=jmin,jmax
                x(k,j)=x(k,j)+dble(ndx)*lbox(k)
              enddo
            endif
          enddo
        enddo
C$OMP end parallel do
      return
      end
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      subroutine setparam(prmtop)
        integer nmax
        parameter(nmax=2000)
        integer natom,ntyp,nbonh,nbona,ntheth,ntheta,nphih,nphia,natyp
        integer nnb,nres,numbnd,numang,nptra,nmol
        double precision charge(nmax),mass(nmax)
        integer ityp(nmax),nbparm(nmax),nrpnt(nmax),nmpnt(nmax)
        double precision kbnd(nmax),rbnd(nmax),kang(nmax),tang(nmax)
        double precision kdih(nmax),pdih(nmax),ndih(nmax)
        double precision scee(nmax),scnb(nmax)
        double precision alj(nmax),blj(nmax)
        integer bhlist(3,nmax),balist(3,nmax)
        integer ahlist(4,nmax),aalist(4,nmax)
        integer dhlist(5,nmax),dalist(5,nmax)
        integer excl(nmax),nexc(nmax)
        integer i,j,inext,i1,i2,i3,i4
        character*64 prmtop
C
        common /param/ charge,mass,kbnd,rbnd,kang,tang,kdih,pdih,ndih,
     &       scee,scnb,alj,blj,ityp,
     &       nbparm,nrpnt,nmpnt,bhlist,balist,ahlist,aalist,dhlist,
     &       dalist,excl,nexc,natom,ntyp,nbonh,nbona,ntheth,
     &       ntheta,nphih,nphia,nnb,nmol,nres
C
        open(20,FILE=prmtop,STATUS='old')
C  numbers
        do i=1,6
          read(20,*)
        enddo
        read(20,999) natom,ntyp,nbonh,nbona,ntheth,ntheta,nphih,nphia
        read(20,998) nnb,nres,numbnd,numang,nptra,natyp
C  charge
        inext=7+(natom-1)/20
        do i=1,inext
          read(20,*)
        enddo
        inext=(natom-1)/5+1
        do i=1,inext
          j=(i-1)*5+1
          read(20,997) charge(j),charge(j+1),charge(j+2),
     &         charge(j+3),charge(j+4)
        enddo
C  mass
        inext=5+(natom-1)/10
        do i=1,inext
          read(20,*)
        enddo
        inext=(natom-1)/5+1
        do i=1,inext
          j=(i-1)*5+1
          read(20,997) mass(j),mass(j+1),mass(j+2),
     &         mass(j+3),mass(j+4)
        enddo
C  ityp
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=(natom-1)/10+1
        do i=1,inext
          j=(i-1)*10+1
          read(20,996) ityp(j),ityp(j+1),ityp(j+2),ityp(j+3),ityp(j+4),
     &         ityp(j+5),ityp(j+6),ityp(j+7),ityp(j+8),ityp(j+9)
        enddo
C  n-excluded
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=(natom-1)/10+1
        do i=1,inext
          j=(i-1)*10+1
          read(20,996) nexc(j),nexc(j+1),nexc(j+2),nexc(j+3),nexc(j+4),
     &         nexc(j+5),nexc(j+6),nexc(j+7),nexc(j+8),nexc(j+9)
        enddo
C
        i2=1
        do i=1,natom
          i1=nexc(i)
          nexc(i)=i2
          i2=i2+i1
        enddo
        nexc(natom+1)=i2
C non-bonded parm
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=(ntyp*ntyp-1)/10+1
        do i=1,inext
          j=(i-1)*10+1
          read(20,996) nbparm(j),nbparm(j+1),nbparm(j+2),nbparm(j+3),
     &         nbparm(j+4),nbparm(j+5),nbparm(j+6),nbparm(j+7),
     &         nbparm(j+8),nbparm(j+9)
        enddo
C  residue pointers
        inext=5+(nres-1)/20
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(nres-1)/10
        do i=1,inext
          j=(i-1)*10+1
          read(20,996) nrpnt(j),nrpnt(j+1),nrpnt(j+2),nrpnt(j+3),
     &         nrpnt(j+4),nrpnt(j+5),nrpnt(j+6),nrpnt(j+7),
     &         nrpnt(j+8),nrpnt(j+9)
        enddo
        nrpnt(nres+1)=natom+1
C  bond force
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(numbnd-1)/5
        do i=1,inext
          j=(i-1)*5+1
          read(20,997) kbnd(j),kbnd(j+1),kbnd(j+2),kbnd(j+3),kbnd(j+4)
        enddo
C  bond distance
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(numbnd-1)/5
        do i=1,inext
          j=(i-1)*5+1
          read(20,997) rbnd(j),rbnd(j+1),rbnd(j+2),rbnd(j+3),rbnd(j+4)
        enddo
C  angle force
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(numang-1)/5
        do i=1,inext
          j=(i-1)*5+1
          read(20,997) kang(j),kang(j+1),kang(j+2),kang(j+3),kang(j+4)
        enddo
C  angle values
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(numang-1)/5
        do i=1,inext
          j=(i-1)*5+1
          read(20,997) tang(j),tang(j+1),tang(j+2),tang(j+3),tang(j+4)
        enddo
C  dihedral force
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(nptra-1)/5
        do i=1,inext
          j=(i-1)*5+1
          read(20,997) kdih(j),kdih(j+1),kdih(j+2),kdih(j+3),kdih(j+4)
        enddo
C  dihedral preriodicity
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(nptra-1)/5
        do i=1,inext
          j=(i-1)*5+1
          read(20,997) ndih(j),ndih(j+1),ndih(j+2),ndih(j+3),ndih(j+4)
        enddo
C  dihedral phase
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(nptra-1)/5
        do i=1,inext
          j=(i-1)*5+1
          read(20,997) pdih(j),pdih(j+1),pdih(j+2),pdih(j+3),pdih(j+4)
        enddo
C  SCEE scale factor
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(nptra-1)/5
        do i=1,inext
          j=(i-1)*5+1
          read(20,997) scee(j),scee(j+1),scee(j+2),scee(j+3),scee(j+4)
        enddo
C  SCNB scale factor
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(nptra-1)/5
        do i=1,inext
          j=(i-1)*5+1
          read(20,997) scnb(j),scnb(j+1),scnb(j+2),scnb(j+3),scnb(j+4)
        enddo
C  Lennard-Jones A
        inext=5+(natyp-1)/5
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(ntyp*(ntyp+1)/2-1)/5
        do i=1,inext
          j=(i-1)*5+1
          read(20,997) alj(j),alj(j+1),alj(j+2),alj(j+3),alj(j+4)
        enddo
        do i=1,(ntyp*(ntyp+1))/2
          alj(i)=alj(i)*12.d0
        enddo
C  Lennard-Jones B
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(ntyp*(ntyp+1)/2-1)/5
        do i=1,inext
          j=(i-1)*5+1
          read(20,997) blj(j),blj(j+1),blj(j+2),blj(j+3),blj(j+4)
        enddo
        do i=1,(ntyp*(ntyp+1))/2
          blj(i)=blj(i)*6.d0
        enddo
C  Bond list w/ H
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(3*nbonh-1)/10
        do i=1,inext
          j=(i-1)*10+3
          i1=mod((j+1),3)
          if (i1.eq.0) i1=3
          i2=i1+1
          if (i2.eq.4) i2=1
          i3=i2+1
          if (i3.eq.4) i3=1
          read(20,996) bhlist(i1,j/3),bhlist(i2,(j+1)/3),
     &         bhlist(i3,(j+2)/3),bhlist(i1,(j+3)/3),bhlist(i2,(j+4)/3),
     &         bhlist(i3,(j+5)/3),bhlist(i1,(j+6)/3),bhlist(i2,(j+7)/3),
     &         bhlist(i3,(j+8)/3),bhlist(i1,(j+9)/3)
        enddo
        do i=1,nbonh
          bhlist(1,i)=bhlist(1,i)/3+1
          bhlist(2,i)=bhlist(2,i)/3+1
        enddo
C  Bond list w/o H
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(3*nbona-1)/10
        do i=1,inext
          j=(i-1)*10+3
          i1=mod((j+1),3)
          if (i1.eq.0) i1=3
          i2=i1+1
          if (i2.eq.4) i2=1
          i3=i2+1
          if (i3.eq.4) i3=1
          read(20,996) balist(i1,j/3),balist(i2,(j+1)/3),
     &         balist(i3,(j+2)/3),balist(i1,(j+3)/3),balist(i2,(j+4)/3),
     &         balist(i3,(j+5)/3),balist(i1,(j+6)/3),balist(i2,(j+7)/3),
     &         balist(i3,(j+8)/3),balist(i1,(j+9)/3)
        enddo
        do i=1,nbona
          balist(1,i)=balist(1,i)/3+1
          balist(2,i)=balist(2,i)/3+1
        enddo
C  Angle list w/ H
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(4*ntheth-1)/10
        do i=1,inext
          j=(i-1)*10+4
          i1=mod((j+1),4)
          if (i1.eq.0) i1=4
          i2=i1+1
          if (i2.eq.5) i2=1
          i3=i2+1
          if (i3.eq.5) i3=1
          i4=i3+1
          if (i4.eq.5) i4=1
          read(20,996) ahlist(i1,j/4),ahlist(i2,(j+1)/4),
     &         ahlist(i3,(j+2)/4),ahlist(i4,(j+3)/4),ahlist(i1,(j+4)/4),
     &         ahlist(i2,(j+5)/4),ahlist(i3,(j+6)/4),ahlist(i4,(j+7)/4),
     &         ahlist(i1,(j+8)/4),ahlist(i2,(j+9)/4)
        enddo
        do i=1,ntheth
          ahlist(1,i)=ahlist(1,i)/3+1
          ahlist(2,i)=ahlist(2,i)/3+1
          ahlist(3,i)=ahlist(3,i)/3+1
        enddo
C  Angle list w/o H
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(4*ntheta-1)/10
        do i=1,inext
          j=(i-1)*10+4
          i1=mod((j+1),4)
          if (i1.eq.0) i1=4
          i2=i1+1
          if (i2.eq.5) i2=1
          i3=i2+1
          if (i3.eq.5) i3=1
          i4=i3+1
          if (i4.eq.5) i4=1
          read(20,996) aalist(i1,j/4),aalist(i2,(j+1)/4),
     &         aalist(i3,(j+2)/4),aalist(i4,(j+3)/4),aalist(i1,(j+4)/4),
     &         aalist(i2,(j+5)/4),aalist(i3,(j+6)/4),aalist(i4,(j+7)/4),
     &         aalist(i1,(j+8)/4),aalist(i2,(j+9)/4)
        enddo
        do i=1,ntheta
          aalist(1,i)=aalist(1,i)/3+1
          aalist(2,i)=aalist(2,i)/3+1
          aalist(3,i)=aalist(3,i)/3+1
        enddo
C  Dihedral list w/ H
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(5*nphih-1)/10
        do i=1,inext
          j=(i-1)*2+1
          read(20,996) dhlist(1,j),dhlist(2,j),
     &         dhlist(3,j),dhlist(4,j),dhlist(5,j),
     &         dhlist(1,j+1),dhlist(2,j+1),dhlist(3,j+1),
     &         dhlist(4,j+1),dhlist(5,j+1)
        enddo
        do i=1,nphih
          dhlist(1,i)=dhlist(1,i)/3+1
          dhlist(2,i)=dhlist(2,i)/3+1
          if (dhlist(3,i).lt.0) then
            dhlist(3,i)=dhlist(3,i)/3-1
          else
            dhlist(3,i)=dhlist(3,i)/3+1
          endif
          if (dhlist(4,i).lt.0) then
            dhlist(4,i)=dhlist(4,i)/3-1
          else
            dhlist(4,i)=dhlist(4,i)/3+1
          endif
        enddo
        do i=2,nphih
          j=i-1
          if (iabs(dhlist(1,i)).eq.iabs(dhlist(1,j))) then
            if (iabs(dhlist(2,i)).eq.iabs(dhlist(2,j))) then
              if (iabs(dhlist(3,i)).eq.iabs(dhlist(3,j))) then
                if (iabs(dhlist(4,i)).eq.iabs(dhlist(4,j))) then
                  dhlist(2,i)=-dhlist(2,i)
                endif
              endif
            endif
          endif
        enddo
C  Dihedral list w/o H
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(5*nphia-1)/10
        do i=1,inext
          j=(i-1)*2+1
          read(20,996) dalist(1,j),dalist(2,j),
     &         dalist(3,j),dalist(4,j),dalist(5,j),
     &         dalist(1,j+1),dalist(2,j+1),dalist(3,j+1),
     &         dalist(4,j+1),dalist(5,j+1)
        enddo
        do i=1,nphia
          dalist(1,i)=dalist(1,i)/3+1
          dalist(2,i)=dalist(2,i)/3+1
          if (dalist(3,i).lt.0) then
            dalist(3,i)=dalist(3,i)/3-1
          else
            dalist(3,i)=dalist(3,i)/3+1
          endif
          if (dalist(4,i).lt.0) then
            dalist(4,i)=dalist(4,i)/3-1
          else
            dalist(4,i)=dalist(4,i)/3+1
          endif
        enddo
        do i=2,nphia
          j=i-1
          if (iabs(dalist(1,i)).eq.iabs(dalist(1,j))) then
            if (iabs(dalist(2,i)).eq.iabs(dalist(2,j))) then
              if (iabs(dalist(3,i)).eq.iabs(dalist(3,j))) then
                if (iabs(dalist(4,i)).eq.iabs(dalist(4,j))) then
                  dalist(2,i)=-dalist(2,i)
                endif
              endif
            endif
          endif
        enddo
C  Excluded atom list
        inext=2
        do i=1,inext
          read(20,*)
        enddo
        inext=1+(nnb-1)/10
        do i=1,inext
          j=(i-1)*10+1
          read(20,996) excl(j),excl(j+1),excl(j+2),excl(j+3),excl(j+4),
     &         excl(j+5),excl(j+6),excl(j+7),excl(j+8),excl(j+9)
        enddo
C
999     format(8(i8))
998     format(2(i8),3(8X),4(i8))
997     format(5(e16.8))
996     format(10(i8))
        close(20)
C
      return
      end
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      subroutine getfrc(x,lbox,hlbox,f)
        include 'omp_lib.h'
        integer nmax,hmax
        parameter(nmax=2000,hmax=120)
C paramtop values
        integer natom,nbonh,nbona,ntheth,ntheta
        integer nphih,nphia,nnb,nres,ntyp,nmol
        double precision charge(nmax),mass(nmax)
        integer ityp(nmax),nbparm(nmax),nrpnt(nmax),nmpnt(nmax)
        double precision kbnd(nmax),rbnd(nmax),kang(nmax),tang(nmax)
        double precision kdih(nmax),pdih(nmax),ndih(nmax)
        double precision scee(nmax),scnb(nmax)
        double precision alj(nmax),blj(nmax)
        integer bhlist(3,nmax),balist(3,nmax)
        integer ahlist(4,nmax),aalist(4,nmax)
        integer dhlist(5,nmax),dalist(5,nmax)
        integer excl(nmax),nexc(nmax)
C solvent values
        integer npts,ncpu,nstyp
        integer styp(nmax),imin(nmax)
        double precision gr(hmax,nmax),Emf(hmax,nmax)
        double precision flj(hmax,nmax),vtot(nmax)
        double precision e0,p0,rmax,eps
C random
        integer ma(55),inext,inextp
C
        double precision x(3,nmax)
        double precision lbox(3),hlbox(3)
C
        integer i,j,k,iex,jex,it,jt,nlj,jmax,n,igr
        integer todo(nmax)
        integer i1,i2,i3,i4,i5
        double precision fbnd,fang,f14e,f14v,fdih,frst1,frst2
        double precision fdir1,fdir2,fdir3
        double precision f(3,nmax,4),dis(3,nmax)
        double precision fij(3,2),fi(3,20,nmax,4)
        double precision r2,r6,th,r(3),d1(3),d2(3),d3(3),rcom(3)
        double precision c11,c12,c13,c22,c23,c33,a,b
        double precision t1,t2,t3,t4,t5,t6,f1,f4
        integer iin(nmax)
        double precision enow(3),e0now(3),pdotr
C
        double precision rnow,x1,x2,x3,y1,y2,y3,dx,dy,dz
        double precision gnow,fs,r0,p1,p2,dr
        double precision ran3
        integer igo,ngo,ir,tid
C
        common /param/ charge,mass,kbnd,rbnd,kang,tang,kdih,pdih,ndih,
     &       scee,scnb,alj,blj,ityp,
     &       nbparm,nrpnt,nmpnt,bhlist,balist,ahlist,aalist,dhlist,
     &       dalist,excl,nexc,natom,ntyp,nbonh,nbona,ntheth,
     &       ntheta,nphih,nphia,nnb,nmol,nres
C
        common /solvent/ gr,Emf,fLJ,vtot,e0,p0,rmax,eps,
     &       imin,styp,npts,ncpu,nstyp
C
        common /random/ ma,inext,inextp
C$OMP threadprivate(/random/)
C
C  Solvent! & Non-bonded
C$OMP parallel default(none) num_threads(ncpu)
C$OMP& private(i,it,j,jt,k,igr,ngo,rnow,x1,x2,x3,y1,y2,y3,r2,gnow,igo
C$OMP&          ,dx,dy,dz,r6,fs,jmax,r0,n,p1,p2,dr,tid,ir
C$OMP&          ,nlj,fdir1,fdir2,fdir3,iex,jex,th,b
C$OMP&          ,dis,fij,todo,fdih,a,f1,f4,t1,t2,t3,t4,t5
C$OMP&          ,t6,d1,d2,d3,c11,c12,c22,c13,c23,c33,f14v,f14e,i1,i2,i3
C$OMP&          ,i4,i5,fang,r,fbnd,iin,pdotr,enow,e0now)
C$OMP& shared(natom,npts,ityp,fi,x,lbox,hlbox
C$OMP&          ,vtot,flj,nres,nrpnt,nexc,excl,ntyp,styp,imin,emf,gr
C$OMP&          ,nbparm,alj,blj,charge,f,ncpu,dalist,nphia,dhlist
C$OMP&          ,nphih,ahlist,pdih,kdih,ndih,scee,scnb,ntheth,tang,kang
C$OMP&          ,aalist,ntheta,bhlist,nbonh,rbnd,kbnd,balist,nbona
C$OMP&          ,mass,frst2,frst1,e0,p0,eps
C$OMP&          ,rmax,rcom)
        tid=omp_get_thread_num()+1
        do i=1,natom
          do k=1,4
            fi(1,tid,i,k)=0.d0
            fi(2,tid,i,k)=0.d0
            fi(3,tid,i,k)=0.d0
          enddo
        enddo
C$OMP do schedule(dynamic)
        do i=1,natom
          iex=nexc(i)
          jex=nexc(i+1)-1
          it=ityp(i)
          do j=1,natom
            x1=x(1,i)-x(1,j)
            if (x1.gt. hlbox(1)) x1=x1-lbox(1)
            if (x1.lt.-hlbox(1)) x1=x1+lbox(1)
            x2=x(2,i)-x(2,j)
            if (x2.gt. hlbox(2)) x2=x2-lbox(2)
            if (x2.lt.-hlbox(2)) x2=x2+lbox(2)
            x3=x(3,i)-x(3,j)
            if (x3.gt. hlbox(3)) x3=x3-lbox(3)
            if (x3.lt.-hlbox(3)) x3=x3+lbox(3)
            dis(1,j)=x1
            dis(2,j)=x2
            dis(3,j)=x3
            if (j.gt.i) then
              if (excl(iex).ne.j) then
                jt=ityp(j)
                r2=x1**2+x2**2+x3**2
                nlj=ntyp*(it-1)+jt
                nlj=nbparm(nlj)
                r6=r2**(-3)
                fdir2=r6*(alj(nlj)*r6-blj(nlj))/r2
                r0=dsqrt(r2)
                if(r0.gt.2.d0*rmax) then
                  fdir3=-charge(i)*charge(j)/r0/r2*2.d0*
     &                 (1.d0-1.d0/eps)/3.d0
                else
                  fdir3=-charge(i)*charge(j)*((1.d0-1.d0/eps)*
     &                 (8.d0*rmax-3.d0*r0)/24.d0/rmax**4)
                endif
                fdir1=charge(i)*charge(j)/r2/r0
              else
                if (iex.ne.jex) iex=iex+1

                r2=x1**2+x2**2+x3**2
                r0=dsqrt(r2)
                if(r0.gt.2.d0*rmax) then
                  fdir3=-charge(i)*charge(j)/r0/r2*
     &                 2.d0/3.d0*(1.d0-1.d0/eps)
                else
                  fdir3=-charge(i)*charge(j)*(1.d0-1.d0/eps)*
     &                 (8.d0*rmax-3.d0*r0)/24.d0/rmax**4
                endif
                fdir1=0.d0
                fdir2=0.d0
              endif
              fi(1,tid,i,1)=fi(1,tid,i,1)+fdir1*x1
              fi(2,tid,i,1)=fi(2,tid,i,1)+fdir1*x2
              fi(3,tid,i,1)=fi(3,tid,i,1)+fdir1*x3
              fi(1,tid,j,1)=fi(1,tid,j,1)-fdir1*x1
              fi(2,tid,j,1)=fi(2,tid,j,1)-fdir1*x2
              fi(3,tid,j,1)=fi(3,tid,j,1)-fdir1*x3
C              
              fi(1,tid,i,2)=fi(1,tid,i,2)+fdir2*x1
              fi(2,tid,i,2)=fi(2,tid,i,2)+fdir2*x2
              fi(3,tid,i,2)=fi(3,tid,i,2)+fdir2*x3
              fi(1,tid,j,2)=fi(1,tid,j,2)-fdir2*x1
              fi(2,tid,j,2)=fi(2,tid,j,2)-fdir2*x2
              fi(3,tid,j,2)=fi(3,tid,j,2)-fdir2*x3
C
              fi(1,tid,i,3)=fi(1,tid,i,3)+fdir3*x1
              fi(2,tid,i,3)=fi(2,tid,i,3)+fdir3*x2
              fi(3,tid,i,3)=fi(3,tid,i,3)+fdir3*x3
              fi(1,tid,j,3)=fi(1,tid,j,3)-fdir3*x1
              fi(2,tid,j,3)=fi(2,tid,j,3)-fdir3*x2
              fi(3,tid,j,3)=fi(3,tid,j,3)-fdir3*x3
            endif
          enddo
C
          do ngo=1,npts
            rnow=rmax*ran3()**(1.d0/3.d0)
            x1=1.d0-2.d0*ran3()
            x2=1.d0-2.d0*ran3()
            r2=x1**2+x2**2
            do while (r2.ge.1.d0)
              x1=1.d0-2.d0*ran3()
              x2=1.d0-2.d0*ran3()
              r2=x1**2+x2**2
            enddo
            y1=rnow*(1.d0-2.d0*r2)
            r2=2.d0*dsqrt(1.d0-r2)
            y2=rnow*x1*r2
            y3=rnow*x2*r2
C
            gnow=0.d0
            enow(1)=0.d0
            enow(2)=0.d0
            enow(3)=0.d0
            e0now(1)=0.d0
            e0now(2)=0.d0
            e0now(3)=0.d0
            igo=0
            igr=1
            do j=1,natom
              jt=styp(j)
              dx=y1+dis(1,j)
              dy=y2+dis(2,j)
              dz=y3+dis(3,j)
              r2=dx**2+dy**2+dz**2
              r0=dsqrt(r2)
              if(r0.lt.rmax) then
                dr=r0/0.1d0+0.5d0
                ir=int(dr)
                dr=dr-dble(ir)
                if (ir.lt.imin(jt)) igr=0
                gnow=gnow+(gr(ir,jt)+dr*(gr(ir+1,jt)-gr(ir,jt)))
                enow(1)=enow(1)+dx/r0*
     &               (Emf(ir,jt)+dr*(Emf(ir+1,jt)-Emf(ir,jt)))
                enow(2)=enow(2)+dy/r0*
     &               (Emf(ir,jt)+dr*(Emf(ir+1,jt)-Emf(ir,jt)))
                enow(3)=enow(3)+dz/r0*
     &               (Emf(ir,jt)+dr*(Emf(ir+1,jt)-Emf(ir,jt)))
                iin(j)=1
                igo=igo+1
              else
                e0now(1)=e0now(1)-e0*charge(j)*dx/r2/r0
                e0now(2)=e0now(2)-e0*charge(j)*dy/r2/r0
                e0now(3)=e0now(3)-e0*charge(j)*dz/r2/r0
                iin(j)=0
              endif
              enow(1)=enow(1)-e0*charge(j)*dx/r2/r0
              enow(2)=enow(2)-e0*charge(j)*dy/r2/r0
              enow(3)=enow(3)-e0*charge(j)*dz/r2/r0
            enddo
C  Change enow into polarization
            r2=enow(1)**2+enow(2)**2+enow(3)**2
            r0=dsqrt(r2)
            r0=(1.d0/dtanh(r0)-1.d0/r0)/r0
            enow(1)=enow(1)*r0
            enow(2)=enow(2)*r0
            enow(3)=enow(3)*r0
            e0now(1)=e0now(1)/3.d0
            e0now(2)=e0now(2)/3.d0
            e0now(3)=e0now(3)/3.d0
            if(igr.eq.1) then
              gnow=dexp(gnow)
            else
              gnow=0.d0
            endif
C
            do j=1,natom
              jt=styp(j)
              dx=y1+dis(1,j)
              dy=y2+dis(2,j)
              dz=y3+dis(3,j)
              r2=dx**2+dy**2+dz**2
              r0=dsqrt(r2)
C  Coulomb
              fs=-charge(j)*p0/r2/r0
              pdotr=3.d0*(enow(1)*dx+enow(2)*dy+enow(3)*dz)/r2
              fij(1,1)=fs*(pdotr*dx-enow(1))
              fij(2,1)=fs*(pdotr*dy-enow(2))
              fij(3,1)=fs*(pdotr*dz-enow(3))
C  LJ
              if(r0.lt.rmax) then
                dr=r0/0.1d0+0.5d0
                ir=int(dr)
                dr=dr-dble(ir)
                fs=-(flj(ir,jt)+dr*(flj(ir+1,jt)-flj(ir,jt)))/r0
                fij(1,2)=fs*dx
                fij(2,2)=fs*dy
                fij(3,2)=fs*dz
              else
                fij(1,2)=0.d0
                fij(2,2)=0.d0
                fij(3,2)=0.d0            
              endif
C
              fij(1,1)=fij(1,1)*gnow
              fij(2,1)=fij(2,1)*gnow
              fij(3,1)=fij(3,1)*gnow
              fij(1,2)=fij(1,2)*gnow
              fij(2,2)=fij(2,2)*gnow
              fij(3,2)=fij(3,2)*gnow
C  Constant dielectric
              if(iin(j).eq.0) then
                pdotr=3.d0*(e0now(1)*dx+e0now(2)*dy+e0now(3)*dz)/r2
                fij(1,1)=fij(1,1)+fs*(pdotr*dx-e0now(1))
                fij(2,1)=fij(2,1)+fs*(pdotr*dy-e0now(2))
                fij(3,1)=fij(3,1)+fs*(pdotr*dz-e0now(3))
              endif
C  Add to forces
              fi(1,tid,j,3)=fi(1,tid,j,3)+fij(1,1)*vtot(it)/dble(igo)
              fi(2,tid,j,3)=fi(2,tid,j,3)+fij(2,1)*vtot(it)/dble(igo)
              fi(3,tid,j,3)=fi(3,tid,j,3)+fij(3,1)*vtot(it)/dble(igo)
C
              fi(1,tid,j,4)=fi(1,tid,j,4)+fij(1,2)*vtot(it)/dble(igo)
              fi(2,tid,j,4)=fi(2,tid,j,4)+fij(2,2)*vtot(it)/dble(igo)
              fi(3,tid,j,4)=fi(3,tid,j,4)+fij(3,2)*vtot(it)/dble(igo)
            enddo
          enddo
        enddo
C$OMP end do
CC  Bonds
CC$OMP do schedule(static)
C        do i=1,nbona
C          i1=balist(1,i)
C          i2=balist(2,i)
C          it=balist(3,i)
C          r2=0.d0
C          do k=1,3
C            r(k)=x(k,i1)-x(k,i2)
C            r2=r2+r(k)*r(k)
C          enddo
C          fbnd=2.d0*kbnd(it)*(rbnd(it)/dsqrt(r2)-1.d0)
C          do k=1,3
C            fi(k,tid,i1)=fi(k,tid,i1)+fbnd*r(k)
C            fi(k,tid,i2)=fi(k,tid,i2)-fbnd*r(k)
C          enddo
C        enddo
CC$OMP end do
CC$OMP do schedule(static)
C        do i=1,nbonh
C          i1=bhlist(1,i)
C          i2=bhlist(2,i)
C          it=bhlist(3,i)
C          r2=0.d0
C          do k=1,3
C            r(k)=x(k,i1)-x(k,i2)
C            r2=r2+r(k)**2
C          enddo
C          fbnd=2.d0*kbnd(it)*(rbnd(it)/dsqrt(r2)-1.d0)
C          do k=1,3
C            fi(k,tid,i1)=fi(k,tid,i1)+fbnd*r(k)
C            fi(k,tid,i2)=fi(k,tid,i2)-fbnd*r(k)
C          enddo
C        enddo
CC$OMP end do
CC  Angles
CC$OMP do schedule(static)
C        do i=1,ntheta
C          i1=aalist(1,i)
C          i2=aalist(2,i)
C          i3=aalist(3,i)
C          it=aalist(4,i)
C          c11=0.d0
C          c22=0.d0
C          c12=0.d0
C          do k=1,3
C            d1(k)=x(k,i1)-x(k,i2)
C            d2(k)=x(k,i2)-x(k,i3)
C            c11=c11+d1(k)*d1(k)
C            c22=c22+d2(k)*d2(k)
C            c12=c12+d1(k)*d2(k)
C          enddo
C          b=-c12/dsqrt(c11*c22)
C          if (b.ge.1.d0) then
C            th=1.d-16
C          elseif (b.le.-1.d0) then
C            th=3.1415926535d0
C          else
C            th=dacos(b)
C          endif
C          fang=2.d0*kang(it)*(th-tang(it))/dsqrt(c11*c22-c12*c12)
C          do k=1,3
C            fi(k,tid,i1)=fi(k,tid,i1)+fang*(c12/c11*d1(k)-d2(k))
C            fi(k,tid,i2)=fi(k,tid,i2)+fang*
C     &           ((1.d0+c12/c22)*d2(k)-(1.d0+c12/c11)*d1(k))
C            fi(k,tid,i3)=fi(k,tid,i3)+fang*(d1(k)-c12/c22*d2(k))
C          enddo
C        enddo
CC$OMP end do
CC$OMP do schedule(static)
C        do i=1,ntheth
C          i1=ahlist(1,i)
C          i2=ahlist(2,i)
C          i3=ahlist(3,i)
C          it=ahlist(4,i)
C          c11=0.d0
C          c22=0.d0
C          c12=0.d0
C          do k=1,3
C            d1(k)=x(k,i1)-x(k,i2)
C            d2(k)=x(k,i2)-x(k,i3)
C            c11=c11+d1(k)*d1(k)
C            c22=c22+d2(k)*d2(k)
C            c12=c12+d1(k)*d2(k)
C          enddo
C          b=-c12/dsqrt(c11*c22)
C          if (b.ge.1.d0) then
C            th=1.d-16
C          elseif (b.le.-1.d0) then
C            th=3.1415926535d0
C          else
C            th=dacos(b)
C          endif
C          fang=2.d0*kang(it)*(th-tang(it))/dsqrt(c11*c22-c12*c12)
C          do k=1,3
C            fi(k,tid,i1)=fi(k,tid,i1)+fang*(c12/c11*d1(k)-d2(k))
C            fi(k,tid,i2)=fi(k,tid,i2)+fang*
C     &           ((1.d0+c12/c22)*d2(k)-(1.d0+c12/c11)*d1(k))
C            fi(k,tid,i3)=fi(k,tid,i3)+fang*(d1(k)-c12/c22*d2(k))
C          enddo
C        enddo
CC$OMP end do
CC  Dihedrals
C$OMP do schedule(static)
        do i=1,nphih
          i1=dhlist(1,i)
          i2=dhlist(2,i)
          i3=dhlist(3,i)
          i4=dhlist(4,i)
          i5=dhlist(5,i)
          if (i4.lt.0) i4=-i4
          if (i3.lt.0) then
            i3=-i3
          else
            r2=0.d0
            do k=1,3
              r(k)=x(k,i1)-x(k,i4)
              r2=r2+r(k)*r(k)
            enddo
            r6=r2**(-3)
            it=ityp(i1)
            jt=ityp(i4)
            nlj=ntyp*(it-1)+jt
            nlj=nbparm(nlj)
            f14e=charge(i1)*charge(i4)/r2/dsqrt(r2)/scee(i5)
            f14v=r6*(alj(nlj)*r6-blj(nlj))/scnb(i5)/r2
            do k=1,3
              fi(k,tid,i1,1)=fi(k,tid,i1,1)+f14e*r(k)
              fi(k,tid,i4,1)=fi(k,tid,i4,1)-f14e*r(k)
              fi(k,tid,i1,2)=fi(k,tid,i1,2)+f14v*r(k)
              fi(k,tid,i4,2)=fi(k,tid,i4,2)-f14v*r(k)
            enddo
          endif
C
C          if (i2.lt.0) i2=-i2
C          c11=0.d0
C          c12=0.d0
C          c13=0.d0
C          c22=0.d0
C          c23=0.d0
C          c33=0.d0
C          do k=1,3
C            d1(k)=x(k,i1)-x(k,i2)
C            d2(k)=x(k,i2)-x(k,i3)
C            d3(k)=x(k,i3)-x(k,i4)
C            c11=c11+d1(k)*d1(k)
C            c12=c12+d1(k)*d2(k)
C            c13=c13+d1(k)*d3(k)
C            c22=c22+d2(k)*d2(k)
C            c23=c23+d2(k)*d3(k)
C            c33=c33+d3(k)*d3(k)
C          enddo
CC
C          t1=c13*c22-c12*c23
C          t2=c11*c23-c12*c13
C          t3=c12*c12-c11*c22
C          t4=c22*c33-c23*c23
C          t5=c13*c23-c12*c33
C          t6=-t1
CC
C          b=dsqrt(-t3*t4)
CC
C          a=t6/b
C          if (a.le.-1.d0) then
C            fdih=0.d0
C          elseif (a.ge.1.d0) then
C            fdih=0.d0
C          else
C            th=dacos(a)
C            fdih=ndih(i5)*kdih(i5)*dsin(ndih(i5)*th-pdih(i5))
C     &           /dsin(th)*c22/b
C          endif
C          do k=1,3
C            f1=fdih*(t1*d1(k)+t2*d2(k)+t3*d3(k))/t3
C            f4=-fdih*(t4*d1(k)+t5*d2(k)+t6*d3(k))/t4
C            fi(k,tid,i1)=fi(k,tid,i1)+f1
C            fi(k,tid,i2)=fi(k,tid,i2)-(1.d0+c12/c22)*f1+c23/c22*f4
C            fi(k,tid,i3)=fi(k,tid,i3)+c12/c22*f1-(1.d0+c23/c22)*f4
C            fi(k,tid,i4)=fi(k,tid,i4)+f4
C          enddo
        enddo
C$OMP end do
C$OMP do schedule(static)
        do i=1,nphia
          i1=dalist(1,i)
          i2=dalist(2,i)
          i3=dalist(3,i)
          i4=dalist(4,i)
          i5=dalist(5,i)
          if (i4.lt.0) i4=-i4
          if (i3.lt.0) then
            i3=-i3
          else
            r2=0.d0
            do k=1,3
              r(k)=x(k,i1)-x(k,i4)
              r2=r2+r(k)*r(k)
            enddo
            r6=r2**(-3)
            it=ityp(i1)
            jt=ityp(i4)
            nlj=ntyp*(it-1)+jt
            nlj=nbparm(nlj)
            f14e=charge(i1)*charge(i4)/r2/dsqrt(r2)/scee(i5)
            f14v=r6*(alj(nlj)*r6-blj(nlj))/scnb(i5)/r2
            do k=1,3
              fi(k,tid,i1,1)=fi(k,tid,i1,1)+f14e*r(k)
              fi(k,tid,i4,1)=fi(k,tid,i4,1)-f14e*r(k)
              fi(k,tid,i1,2)=fi(k,tid,i1,2)+f14v*r(k)
              fi(k,tid,i4,2)=fi(k,tid,i4,2)-f14v*r(k)
            enddo
          endif
C
C          if (i2.lt.0) i2=-i2
C          c11=0.d0
C          c12=0.d0
C          c13=0.d0
C          c22=0.d0
C          c23=0.d0
C          c33=0.d0
C          do k=1,3
C            d1(k)=x(k,i1)-x(k,i2)
C            d2(k)=x(k,i2)-x(k,i3)
C            d3(k)=x(k,i3)-x(k,i4)
C            c11=c11+d1(k)*d1(k)
C            c12=c12+d1(k)*d2(k)
C            c13=c13+d1(k)*d3(k)
C            c22=c22+d2(k)*d2(k)
C            c23=c23+d2(k)*d3(k)
C            c33=c33+d3(k)*d3(k)
C          enddo
CC
C          t1=c13*c22-c12*c23
C          t2=c11*c23-c12*c13
C          t3=c12*c12-c11*c22
C          t4=c22*c33-c23*c23
C          t5=c13*c23-c12*c33
C          t6=-t1
CC
C          b=dsqrt(-t3*t4)
CC
C          a=t6/b
C          if (a.le.-1.d0) then
C            fdih=0.d0
C          elseif (a.ge.1.d0) then
C            fdih=0.d0
C          else
C            th=dacos(a)
C            fdih=ndih(i5)*kdih(i5)*dsin(ndih(i5)*th-pdih(i5))
C     &           /dsin(th)*c22/b
C          endif
C          do k=1,3
C            f1=fdih*(t1*d1(k)+t2*d2(k)+t3*d3(k))/t3
C            f4=-fdih*(t4*d1(k)+t5*d2(k)+t6*d3(k))/t4
C            fi(k,tid,i1)=fi(k,tid,i1)+f1
C            fi(k,tid,i2)=fi(k,tid,i2)-(1.d0+c12/c22)*f1+c23/c22*f4
C            fi(k,tid,i3)=fi(k,tid,i3)+c12/c22*f1-(1.d0+c23/c22)*f4
C            fi(k,tid,i4)=fi(k,tid,i4)+f4
C          enddo
        enddo
C$OMP end do
C$OMP do schedule(static)
        do i=1,natom
          do k=1,4
            f(1,i,k)=0.d0
            f(2,i,k)=0.d0
            f(3,i,k)=0.d0
          enddo
          do k=1,4
            do j=1,ncpu
              f(1,i,k)=f(1,i,k)+fi(1,j,i,k)
              f(2,i,k)=f(2,i,k)+fi(2,j,i,k)
              f(3,i,k)=f(3,i,k)+fi(3,j,i,k)
            enddo
          enddo
        enddo
C$OMP end do
C$OMP end parallel
C
      return
      end
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      subroutine setsolvent(charge)
        integer nmax,hmax
        parameter(nmax=2000,hmax=120)
C solvent values
        integer npts,ncpu,nstyp
        integer styp(nmax),ttyp(nmax),imin(nmax)
        double precision gr(hmax,nmax),Emf(hmax,nmax)
        double precision fLJ(hmax,nmax),vtot(nmax)
        double precision e0,p0,rmax,eps
C
        double precision charge(nmax)
C
        integer i,j,k,it
        double precision g0,r0
        double precision rho,pi
C     
        common /solvent/ gr,Emf,fLJ,vtot,e0,p0,rmax,eps,
     &       imin,styp,npts,ncpu,nstyp
C
        npts=1000000
        rmax=(dble(hmax)-0.5d0)*0.1d0
C
        pi=3.1415926536d0
        eps=4.39d0
        rho=0.0075d0
        p0=0.2915d0*dsqrt(332.d0)
        e0=3.d0/4.d0/pi/rho/p0*(1.d0-1.d0/eps)
C

        open(20,FILE='LO.crd',STATUS='old')
        read(20,*) it,nstyp
        do i=1,it
          read(20,*) j
          styp(i)=j
          styp(i+it)=j
          ttyp(j)=i
        enddo
        close(20)
C
        open(20,FILE='LO.d0.00.SPA2.rc12.sym.dat',STATUS='old')
        do i=1,nstyp
          do j=1,hmax
            read(20,*) r0,g0,it,k
            if(k.le.0) then
              imin(i)=j+2
            endif
            gr(j,i)=g0
          enddo
        enddo
        close(20)
C
        open(20,FILE='LO.d0.00.pol2.rc12.sym.dat',STATUS='old')
        do i=1,nstyp
          it=ttyp(i)
          do j=1,hmax
            read(20,*) r0,g0
            Emf(j,i)=g0
          enddo
        enddo
        close(20)
C     
        open(20,FILE='LO.d0.00.frc.rc12.sym.dat',STATUS='old')
        do i=1,nstyp
          do j=1,hmax
            read(20,*) r0,g0
            fLJ(j,i)=g0
          enddo
        enddo
        close(20)
C     
        do i=1,nstyp
         vtot(i)=4.d0/3.d0*pi*rmax**3/dble(npts)*rho
        enddo
C     
      return
      end
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      subroutine getinput(prmtop,frst)
        integer nmax
        parameter(nmax=2000)
        character*64 frst,prmtop
        integer i
C
        open(20,FILE="frc.inp",STATUS='old')
        read(20,999) prmtop
        read(20,999) frst
999     format(a)
        close(20)
C
      return
      end
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      subroutine makemol()
        integer nmax
        parameter(nmax=2000)
C
        integer ires(nmax),imol(nmax),neigh(10,1000)
        integer todo(nmax)
        integer i,inow,i1,i2,j1,j2,n,k,j
C paramtop values
        integer natom,nbonh,nbona,ntheth,ntheta
        integer nphih,nphia,nnb,nres,ntyp,nmol
        double precision charge(nmax),mass(nmax)
        integer ityp(nmax),nbparm(nmax),nrpnt(nmax),nmpnt(nmax)
        double precision kbnd(nmax),rbnd(nmax),kang(nmax),tang(nmax)
        double precision kdih(nmax),pdih(nmax),ndih(nmax)
        double precision scee(nmax),scnb(nmax)
        double precision alj(nmax),blj(nmax)
        integer bhlist(3,nmax),balist(3,nmax)
        integer ahlist(4,nmax),aalist(4,nmax)
        integer dhlist(5,nmax),dalist(5,nmax)
        integer excl(nmax),nexc(nmax)
C
        common /param/ charge,mass,kbnd,rbnd,kang,tang,kdih,pdih,ndih,
     &       scee,scnb,alj,blj,ityp,
     &       nbparm,nrpnt,nmpnt,bhlist,balist,ahlist,aalist,dhlist,
     &       dalist,excl,nexc,natom,ntyp,nbonh,nbona,ntheth,
     &       ntheta,nphih,nphia,nnb,nmol,nres
C
        do i=1,nres
          neigh(1,i)=1
        enddo
C
        inow=0
        do i=1,natom
          if (i.eq.nrpnt(inow+1)) inow=inow+1
          ires(i)=inow
        enddo
        do i=1,nbonh
          i1=bhlist(1,i)
          i2=bhlist(2,i)
          j1=ires(i1)
          j2=ires(i2)
          if (j1.ne.j2) then
            n=neigh(1,j1)+1
            neigh(1,j1)=n
            neigh(n,j1)=j2
            n=neigh(1,j2)+1
            neigh(1,j2)=n
            neigh(n,j2)=j1
          endif
        enddo
        do i=1,nbona
          i1=balist(1,i)
          i2=balist(2,i)
          j1=ires(i1)
          j2=ires(i2)
          if (j1.ne.j2) then
            n=neigh(1,j1)+1
            neigh(1,j1)=n
            neigh(n,j1)=j2
            n=neigh(1,j2)+1
            neigh(1,j2)=n
            neigh(n,j2)=j1
          endif
        enddo
C
        nmol=0
        do i=1,nres
          imol(i)=0
        enddo
C
        do i=1,nres
          if (imol(i).eq.0) then
            nmol=nmol+1
            imol(i)=nmol
            i1=1
            i2=1
            todo(1)=i
            do while (i1.le.i2)
              j=todo(i1)
              j1=neigh(1,j)
              do k=2,j1
                j2=neigh(k,j)
                if (imol(j2).eq.0) then
                  i2=i2+1
                  todo(i2)=j2
                  imol(j2)=nmol
                endif
              enddo
              i1=i1+1
            enddo
          endif
        enddo
C
        j=0
        do i=1,nres
          if (imol(i).ne.j) then
            if (j+1.ne.imol(i)) then
              write(0,*) 'NONE SEQUENTIAL RESIDUES IN MOLECULE'
            endif
            j=j+1
            nmpnt(j)=nrpnt(i)
          endif
        enddo
        nmpnt(nmol+1)=natom+1
C
      return
      end
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

! This code is to generate lammps eam/fs pair style file
! output file : **.eam.fs
! writen by Yangchun Chen (ychchen@hnu.edu.cn)  2019-9-16

      program main
      implicit real*8(a-h,o-z)
      dimension emb(10000),fr(10000),vsum(10000),v(10000)
      dimension T_ao(10),T_ro(10),T_ap(10),T_rp(10)
      dimension bzb(4)

!******* ZBL connect parameters     
      qele1 = 12.0d0  !for Mg
      qele2 = 12.0d0  !for Mg
      b0 =  0.704664243977151d+01
      b1 = -0.333010469851498d+01
      b2 = -0.619555583046675d-01
      b3 =  0.640655473828010d-01
!******* Guage parameters
      T_sss = 0.644973966426141d+00
      T_ccc = 0.401551356125883d+00 
!******* potential parameters
      T_ao(1)=0.103505961304011d-02
      T_ao(2)=-0.200456259837736d-02
      T_ao(3)=0.582304132777502d-01
      T_ao(4)=-0.114559033299128d+00
      T_ao(5)=0.572667170534239d-01
      T_ao(6)=0.131921246010663d-03
      T_ao(7)=-0.210012600486547d-03
      T_ao(8)=0.175729529142241d+00
      T_ao(9)=-0.370683599716195d+00
      T_ao(10)=0.214070465439199d+01

      T_ap(1)=-0.259183792597927d-03
      T_ap(2)=-0.904891038338592d-02
      T_ap(3)=0.218862253193047d-01
      T_ap(4)=0.586212126710029d-02
      T_ap(5)=-0.627354410264580d-01
      T_ap(6)=-0.409394774379268d-01
      T_ap(7)=0.282588362952282d+00
      T_ap(8)=0.297152769812915d+00
      T_ap(9)=0.566647129926957d+00
      T_ap(10)=0.981038501882758d-01 

      T_ro(1) = 6.3d0
      T_ro(2) = 5.9d0
      T_ro(3) = 5.5d0
      T_ro(4) = 5.1d0
      T_ro(5) = 4.7d0
      T_ro(6) = 4.3d0
      T_ro(7) = 3.9d0
      T_ro(8) = 3.5d0
      T_ro(9) = 3.1d0
      T_ro(10)= 2.7d0

      T_rp(1) = 6.3d0
      T_rp(2) = 5.9d0
      T_rp(3) = 5.5d0
      T_rp(4) = 5.1d0
      T_rp(5) = 4.7d0
      T_rp(6) = 4.3d0
      T_rp(7) = 3.9d0
      T_rp(8) = 3.5d0
      T_rp(9) = 3.1d0
      T_rp(10)= 2.7d0
!************* 
      dp=2.0000d-0003
      dr=6.3000d-0004
!*************
      open(unit=1,file='Mg_20190916.eam.fs',status='replace')
      write(1,*)"fitting Mg"
      write(1,*)"Finnis-Sinclair formalism"
      write(1,*)"by chenyangchun 2019/9/16"
      write(1,*)"1  Mg"
      write(1,*)"10000  2.0000E-0003  10000  6.3000E-0004  6.3000E+0000"
      write(1,*)"12   2.43050E+0001   3.2090E+0000  hcp"
!**********
      do i=1,10000
      r=float(i-1)*dp   
      if(i==1)then
      emb(i)=0.0d0
      else
      emb(i) = -sqrt(r/T_sss)+T_ccc/T_sss*r
      endif
      end do
      do i=1,2000
      write(1,99)emb(5*i-4),emb(5*i-3),emb(5*i-2),emb(5*i-1),emb(5*i)
      end do
!**********
      do i=1,10000
      r=float(i-1)*dr 
      fr1 =   T_ao(1)*(T_ro(1)-r)**3*HH(T_ro(1)-r)+   
     &        T_ao(2)*(T_ro(2)-r)**3*HH(T_ro(2)-r)+   
     &        T_ao(3)*(T_ro(3)-r)**3*HH(T_ro(3)-r)+   
     &        T_ao(4)*(T_ro(4)-r)**3*HH(T_ro(4)-r)+
     &        T_ao(5)*(T_ro(5)-r)**3*HH(T_ro(5)-r)+   
     &        T_ao(6)*(T_ro(6)-r)**3*HH(T_ro(6)-r)+
     &        T_ao(7)*(T_ro(7)-r)**3*HH(T_ro(7)-r)+   
     &        T_ao(8)*(T_ro(8)-r)**3*HH(T_ro(8)-r)+
     &        T_ao(9)*(T_ro(9)-r)**3*HH(T_ro(9)-r)+   
     &        T_ao(10)*(T_ro(10)-r)**3*HH(T_ro(10)-r)
      fr(i) = T_sss*fr1
      end do
      do i=1,2000
      write(1,99)fr(5*i-4),fr(5*i-3),fr(5*i-2),fr(5*i-1),fr(5*i)
      end do
!*********
      do i=1,10000
      r=float(i-1)*dr
      IF(i==1)THEN
      zed1    = qele1
      zed2    = qele2
      ev      = 1.602176565d-19
      pi      = 3.14159265358979324d0
      epsil0  = 8.854187817d-12
      bohrad  = 0.52917721067d0
      exn     = 0.23d0     
      beta    = (zed1*zed2*ev*ev)/(4.0d0*pi*epsil0)*1.0d10/ev
      rs      = 0.8854d0*bohrad/(zed1**exn +zed2**exn)
      bzb(1) = -3.19980d0/rs
      bzb(2) = -0.94229d0/rs
      bzb(3) = -0.40290d0/rs
      bzb(4) = -0.20162d0/rs
      v(i)  =  beta*(0.18175d0*exp(bzb(1)*r)+
     &               0.50986d0*exp(bzb(2)*r)+
     &               0.28022d0*exp(bzb(3)*r)+
     &               0.02817d0*exp(bzb(4)*r))
      cycle
      ENDIF 
      if(r.lt.1.0d0)then
      zed1    = qele1     
      zed2    = qele2     
      ev      = 1.602176565d-19
      pi      = 3.14159265358979324d0
      epsil0  = 8.854187817d-12
      bohrad  = 0.52917721067d0
      exn     = 0.23d0
      beta    = (zed1*zed2*ev*ev)/(4.0d0*pi*epsil0)*1.0d10/ev
      rs      = 0.8854d0*bohrad/(zed1**exn +zed2**exn)
      bzb(1) = -3.19980d0/rs
      bzb(2) = -0.94229d0/rs
      bzb(3) = -0.40290d0/rs
      bzb(4) = -0.20162d0/rs
      rinv   = 1.0d0/r    
      vsum(i)  = beta*rinv*(0.18175d0*exp(bzb(1)*r)+
     &                      0.50986d0*exp(bzb(2)*r)+
     &                      0.28022d0*exp(bzb(3)*r)+
     &                      0.02817d0*exp(bzb(4)*r))
      elseif((r.ge.1.0d0).and.(r.lt.2.3d0))then
      vsum(i)=exp(b0+b1*r+b2*r**2.0+b3*r**3.0)
      elseif(r.ge.2.3d0)then
      vsum(i) = T_ap(1)*(T_rp(1)-r)**3*HH(T_rp(1)-r)+    
     &          T_ap(2)*(T_rp(2)-r)**3*HH(T_rp(2)-r)+    
     &          T_ap(3)*(T_rp(3)-r)**3*HH(T_rp(3)-r)+     
     &          T_ap(4)*(T_rp(4)-r)**3*HH(T_rp(4)-r)+     
     &          T_ap(5)*(T_rp(5)-r)**3*HH(T_rp(5)-r)+     
     &          T_ap(6)*(T_rp(6)-r)**3*HH(T_rp(6)-r)+
     &          T_ap(7)*(T_rp(7)-r)**3*HH(T_rp(7)-r)+    
     &          T_ap(8)*(T_rp(8)-r)**3*HH(T_rp(8)-r)+    
     &          T_ap(9)*(T_rp(9)-r)**3*HH(T_rp(9)-r)+     
     &          T_ap(10)*(T_rp(10)-r)**3*HH(T_rp(10)-r)
     &   -2.0d0*T_ccc*(T_ao(1)*(T_ro(1)-r)**3*HH(T_ro(1)-r)+   
     &                 T_ao(2)*(T_ro(2)-r)**3*HH(T_ro(2)-r)+   
     &                 T_ao(3)*(T_ro(3)-r)**3*HH(T_ro(3)-r)+   
     &                 T_ao(4)*(T_ro(4)-r)**3*HH(T_ro(4)-r)+
     &                 T_ao(5)*(T_ro(5)-r)**3*HH(T_ro(5)-r)+   
     &                 T_ao(6)*(T_ro(6)-r)**3*HH(T_ro(6)-r)+
     &                 T_ao(7)*(T_ro(7)-r)**3*HH(T_ro(7)-r)+   
     &                 T_ao(8)*(T_ro(8)-r)**3*HH(T_ro(8)-r)+
     &                 T_ao(9)*(T_ro(9)-r)**3*HH(T_ro(9)-r)+   
     &                 T_ao(10)*(T_ro(10)-r)**3*HH(T_ro(10)-r))
      endif
      v(i)=r*vsum(i)
      end do
      do i=1,2000
      write(1,99)v(5*i-4),v(5*i-3),v(5*i-2),v(5*i-1),v(5*i)
      end do
!*************
99    format(5ES25.14E4)
      close(1)

      stop
      end program
!*************
      real*8 function HH(x)
      implicit real*8(a-h,o-z)
      if(x.gt.0.)then
      HH=1.
      else
      HH=0.
      endif
      return
      end function

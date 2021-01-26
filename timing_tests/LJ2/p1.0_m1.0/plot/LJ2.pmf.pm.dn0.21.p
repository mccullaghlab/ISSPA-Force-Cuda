set output "test.tex"
set terminal epslatex

set grid front
set size 0.8
 
set xtic 1

eps=2.35

#eps=4.39
rc=24.95

set multiplot

set origin 0., 0.

set xr [0:15]
set yr [-35:5]

unset label
set label "$R$ (\\AA)" at graph 0.85, graph -0.15
set label "$u_{IS-SPA}$ (kcal/mol)" at graph -0.14, graph 0.55 rotate by 90
set label "\\color{white}." at graph -0.15, graph 0.
LABEL = "\\Large{(a)}"
set obj 10 rect at graph 0.05,0.94 size char 3, char 1.5
set obj 10 front fs noborder
set label 10 at graph 0.05, 0.935 LABEL front center


uC(x)=-332.*(1.-1./eps)*(x**2/6./rc**3-5.*x**3/96./rc**4+3.*x/16./rc**2+1./8./(x+rc)+log(1.+x/rc)/8./x-13./12.)
uLJ(x)=0.152*(7./x)**6*((7./x)**6-2.)

set key at graph 1.0,0.41 width -7

plot 0 lc rgb "black" lw 2 not,\
   uLJ(x)-332./2.35/x lc rgb "black" lw 3 dt 1 t 'CDD + LJ',\
   'decomposed_z1.0_r7.0.out' u 1:($6+$7+uLJ($1)-332./$1+12.05)           w l lw 3 dt 2 lc 1 t '$|q|=1$ e -- explicit',\
   'decomposed_z0.0_r7.0.out' u 1:($6+$7+uLJ($1))           w l lw 3 dt 2 lc 3 t '$q=0$ e -- explicit',\
   'LJ2.q1.00.IS-SPA.dn0.21.dat'  u 1:(($6+$8)+uC($1)+uLJ($1)-332./$1-196.03) w l lw 3 dt 1 lc rgb 'blue' t '$q=+1$ e -- IS-SPA',\
   'LJ2.q1.00.IS-SPA.dn0.21.dat'  u 1:(($7+$9)+uC($1)+uLJ($1)-332./$1-196.85) w l lw 3 dt 1 lc rgb 'red' t '$q=-1$ e -- IS-SPA',\
   'LJ2.q0.00.IS-SPA.dn0.21.dat'  u 1:($6+uLJ($1)) w l lw 3 dt 1 lc 3 t '$q=0$ e -- IS-SPA'

set origin 0.78, 0.

set xr [0:15]
set yr [-4:8]

unset label
set label "$R$ (\\AA)" at graph 0.85, graph -0.15
set label "$\\delta u_{IS-SPA}^{LJ}$ (kcal/mol)" at graph -0.1, graph 0.5 rotate by 90
set label "\\color{white}." at graph -0.15, graph 0.
LABEL = "\\Large{(b)}"
set obj 10 rect at graph 0.05,0.94 size char 3, char 1.5
set obj 10 front fs noborder
set label 10 at graph 0.05, 0.935 LABEL front center

uC(x)=-332.*(1.-1./eps)*(x**2/6./rc**3-5.*x**3/96./rc**4+3.*x/16./rc**2+1./8./(x+rc)+log(1.+x/rc)/8./x-13./12.)
uLJ(x)=0.152*(7./x)**6*((7./x)**6-2.)

set key at graph 1.0,0.98 width -8

plot 0 lc rgb "black" lw 2 not,\
   uLJ(x) lc rgb "black" lw 3 dt 1 t 'LJ',\
   'decomposed_z1.0_r7.0.out' u 1:( $6+uLJ($1))   w l lw 3 dt 2 lc rgb "blue" t '$q=+1$ e -- explicit',\
   'decomposed_z1.0_r7.0.out' u 1:(-$8+uLJ($1))   w l lw 3 dt 2 lc rgb "red" t '$q=-1$ e -- explicit',\
   'decomposed_z0.0_r7.0.out' u 1:($6+$7+uLJ($1)) w l lw 3 dt 2 lc 3 t '$|q|=0$ e -- explicit LJ',\
   'LJ2.q1.00.IS-SPA.dn0.21.dat'  u 1:($6+uLJ($1))     w l lw 3 dt 1 lc rgb 'blue' t '$q=+1$ e -- IS-SPA',\
   'LJ2.q1.00.IS-SPA.dn0.21.dat'  u 1:($7+uLJ($1))     w l lw 3 dt 1 lc rgb 'red' t '$q=-1$ e -- IS-SPA',\
   'LJ2.q0.00.IS-SPA.dn0.21.dat'  u 1:($6+uLJ($1))  w l lw 3 dt 1 lc 3 t '$q=0$ e -- IS-SPA'

set origin 1.57, 0.

set xr [0:15]
set yr [-50:0]

unset label
set label "$R$ (\\AA)" at graph 0.85, graph -0.15
set label "$\\delta u_{pmf}^{C}$ (kcal/mol)" at graph -0.14, graph 0.6 rotate by 90
set label "\\color{white}." at graph -0.15, graph 0.
LABEL = "\\Large{(c)}"
set obj 10 rect at graph 0.05,0.94 size char 3, char 1.5
set obj 10 front fs noborder
set label 10 at graph 0.05, 0.935 LABEL front center


uC(x)=-332.*(1.-1./eps)*(x**2/6./rc**3-5.*x**3/96./rc**4+3.*x/16./rc**2+1./8./(x+rc)+log(1.+x/rc)/8./x-13./12.)
uLJ(x)=0.152*(7./x)**6*((7./x)**6-2.)

set key at graph 1.0,0.35 width -6

plot -332./2.35/x lc rgb "black" lw 3 dt 1 t 'CDD',\
   'decomposed_z1.0_r7.0.out' u 1:( $7-332./$1+12.1) w l lw 3 dt 2 lc rgb "blue" t '$q=+1$ e -- explicit',\
   'decomposed_z1.0_r7.0.out' u 1:(-$9-332./$1+12.0) w l lw 3 dt 2 lc rgb "red" t '$q=-1$ e -- explicit',\
   'LJ2.q1.00.IS-SPA.dn0.21.dat'  u 1:($8+uC($1)-332./$1-196.03)  w l lw 3 dt 1 lc rgb 'blue' t '$q=+1$ e -- IS-SPA',\
   'LJ2.q1.00.IS-SPA.dn0.21.dat'  u 1:($9+uC($1)-332./$1-196.85)  w l lw 3 dt 1 lc rgb 'red' t '$q=-1$ e -- IS-SPA'

unset multiplot

set terminal x11


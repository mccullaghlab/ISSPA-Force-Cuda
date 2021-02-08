set output "test.tex"
set terminal epslatex

set grid front
set size 0.8

set label "$r$ (\\AA)" at graph 0.7, graph -0.15
set label "$f_{solv}^C$ (kcal/mol/\\AA)" at graph -0.1, graph 0.55 rotate by 90
set label "\\color{white}." at graph -0.15, graph 0.

set xr [0:15]
set yr [0:5]

set xtic 1

eps=2.35

#eps=4.39
rc=24.95

fC(x)=332.*(1.-1./eps)*(x*(8.*rc-3.*x)/24./rc**4-log(1.+x/rc)/8./x**2+rc/8./x/(x+rc)**2-x**2/32./rc**4+3./16./rc**2)

plot '../LJ2.qp0.50.qm1.00.IS-SPA.dn0.21.dat' u 1:($4+0.5*fC($1)) w l lw 3 dt 1 lc rgb 'blue' t '$+0.5$ ion - IS-SPA',\
   '' u 1:($5+fC($1)) w l lw 3 dt 1 lc rgb 'red'  t '$-1.0$ ion - IS-SPA',\
   332.*0.5*(1.-1./2.35)/x**2 lc rgb "black" lw 3 dt 1 t 'CDD - $\epsilon = 2.35$'

set terminal x11


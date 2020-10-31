# Weakly Infeasible Semidefinite Program Algorithm

A semidefinite program (SDP) is a class of optimization problem of the form

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\begin{array}{rccl}&space;\inf&space;&&space;C\bullet&space;X&space;\\&space;s.t.&space;&&space;\mathcal{A}(X)&space;&&space;=&space;&&space;b\\&space;&&space;X&space;&&space;\succeq&space;&&space;0&space;\end{array}"  /></p>

for which <a><img src="https://latex.codecogs.com/svg.latex?\bullet"  /></a> is the standard Euclidean inner product between real symmetric matrices, <a><img src="https://latex.codecogs.com/svg.latex?X\succeq0"  /></a> says that <a><img src="https://latex.codecogs.com/svg.latex?X"  /></a> is positive semidefinite and

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\mathcal{A}:X\mapsto\begin{pmatrix}&space;A_1\bullet&space;X&space;\\&space;\vdots&space;\\&space;A_m\bullet&space;X&space;\\&space;\end{pmatrix}"  /></p>

We will refer to the SDP of the above form as the primal problem <a><img src="https://latex.codecogs.com/svg.latex?(P)"  /></a>. If <a><img src="https://latex.codecogs.com/svg.latex?X_0" /></a> is a solution to the above linear system, then we note that the feasible set of this problem may be expressed as the intersection of an affine space

<p align="center"><img src="https://latex.codecogs.com/svg.latex?H:=X_0&plus;\mathcal{N}(\mathcal{A})" /></p>

with the semidefinite cone <a><img src="https://latex.codecogs.com/svg.latex?S^n_&plus;"  /></a>. Thus, we say the problem is *infeasible* if

<p align="center"><img src="https://latex.codecogs.com/svg.latex?H\cap&space;S^n_&plus;=\emptyset"  /></p>

It is *strongly infeasible* if the distance between these sets is positive, that is...

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\mathrm{dist}(H,S^n_&plus;)>0" /></p>

Otherwise, if the problem is *infeasible* but not *strongly infeasible* we say that the problem is *weakly infeasible*.

A related notion is the idea of a bad projection of the semidefinite cone. We say that <a><img src="https://latex.codecogs.com/svg.latex?\mathcal{A}"  /></a> is a *bad projection* of the semidefinite cone if it maps the semidefinite cone to a non-closed set. It is easy to check that

<p align="center"><img src="https://latex.codecogs.com/svg.latex?(P)\text{&space;is&space;weakly&space;infeasible&space;for&space;some&space;}b\iff\mathcal{A}\text{&space;is&space;a&space;bad&space;projection&space;of&space;}S^n_&plus;"  /></p>

Understanding when a linear operator is a bad projection is important in the theory and solution methods for conic linear systems. In particular, it is well known that the closure of a cone under a linear operator is a sufficient condition to guarantee the application of [Farka's Lemma](https://en.wikipedia.org/wiki/Farkas%27_lemma), but one may ask, "how does weak infeasibility appear in theory and application of SDPs?"

Classically, weakly infeasible SDPs were simply viewed as instances in which <img src="https://latex.codecogs.com/svg.latex?H" /> was an asymptote of the semidefinite cone; however, in modern optimization, these instances appear...

* as hard SDPs, identified as feasible even by state-of-the-art solvers such as Mosek or SDPA-GMP. [(Liu, Pataki 2017)](https://arxiv.org/abs/1507.00290)
* as infeasible and ill-posed SDPs with no convergence guarantees on classifying feasibility with [IPMs](https://en.wikipedia.org/wiki/Interior-point_method) based around condition metrics. [(Pena, Renegar 2000)](https://link.springer.com/article/10.1007/s101070050001)
* as bad projections of the semidefinite cone with algebraic characterizations of related bad subspaces. [(Jiang, Sturmfels 2020)](https://arxiv.org/abs/2006.09956)
* as instances in the Lasserre hierarchy of polynomial optimization for certain classes of polynomial optimization problems. [(Henrion, Lasserre 2005)](https://link.springer.com/chapter/10.1007/10997703_15)
* as SDPs in which facial reduction certificates are necessary to certify infeasibility. [(Lourenco, Muramatsu, Tsuchiya 2015)](https://arxiv.org/abs/1507.06843)
* as difficult to generate test cases for SDP solvers. [(Waki 2011)](http://www.optimization-online.org/DB_HTML/2011/07/3086.html)

The code base presented here gives us an algorithm for solving the problem presented in the last of these bullets in its entirety by producing suitable matrices <a><img src="https://latex.codecogs.com/svg.latex?A_1,\dots,A_m" /></a> and right hand side vector <a><img src="https://latex.codecogs.com/svg.latex?b"  /></a>.

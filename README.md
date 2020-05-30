# SBSA_QP

## A MATLAB package for the solution of optimization problems modeling sparse data recovery with fused-lasso regularization.

## Authors
Valentina De Simone, University of Campania "Luigi Vanvitelli", Caserta, Italy, valentina[dot]desimone[at]unicampania[dot]it
Daniela di Serafino, University of Campania "Luigi Vanvitelli", Caserta, Italy, daniela[dot]diserafino[at]unicampania[dot]it   
Marco Viola, University of Campania "Luigi Vanvitelli", Caserta, Italy, marco[dot]viola[at]unicampania[dot]it

## Last Update
Version 1.0 - May 28, 2020

## Description
SBSA_QP is a MATLAB implementation of the "Split Bregman method with Subspace 
Acceleration" (SBSA), proposed in [1] for the solution of constrained
optimization problems of the form

              min  f(u) + tau_1*||u||_1 + tau_2*||D*u||_1
              s.t. A*u = b,

modeling, e.g., multiperiod portfolio optimization problems, regularized
least-squares regression problems or source detection problems in
electroencephalography.

Here f(u) is a quadratic function, having either the form 

             f(u) = 0.5*u'*Q*u - p'*u,

or the least squares form

             f(u) = 0.5*||Q*u - p||^2.

SBSA is an iterative method based on the Split Bregman method
[T. Goldstein and S. Osher, SIAM J. Imaging Sciences, 2 (2009), pp.323-343]
with the introduction of subspace-acceleration steps.
The subspaces where the acceleration is performed are selected so that 
the restriction of the objective function is a smooth function in a 
neighborhood of the current iterate. We refer to the non-accelerated steps
as "standard Bregman steps".

The minimization in standard Bregman steps is performed by using the FISTA
method implemented in the [FOM package](https://sites.google.com/site/fomsolver/)
[A. Beck and N. Guttmann-Beck, Optimization Methods and Software, 34:1 (2019), pp. 172-193]. 
The minimization of the unconstrained quadratic problems in the accelerated
steps is performed by using the CG method.

The algorithm stops if the violation of the linear constraints is below a
given threshold, namely

             ||A*u-b|| <= tol_B    and    ||D*u-d|| <= tol_B,

where d is the auxiliary variable introduced in the Bregman scheme (see
Section 3 in [1] for further details).

### References
[1] V. De Simone, D. di Serafino and M. Viola,
*A subspace-accelerated split Bregman method for sparse data recovery with joint l_1-type regularizers*,
Electronic Transactions on Numerical Analysis (ETNA), volume 53, 2020, pp. 406-425, DOI: 10.1553/etna_vol53s406.
Preprint available from [ArXiv](https://arxiv.org/abs/1912.06805) and [Optimization Online](http://www.optimization-online.org/DB_HTML/2019/12/7519.html).

## Software requirements
SBSA_QP runs under MATLAB. It has been tested under MATLAB 2018b.
SBSA_QP requires the [FOM package](https://sites.google.com/site/fomsolver/);
note that the path to FOM must be added to the MATLAB path with the command

`addpath(genpath('path_to_fom'))`


## Contents of the package
Here's the list of SBSA_QP files.

MAIN FILES:

- `sbsa_qp.m`    : main function;
- `CGsubspace.m` : CG method for solving subspace-acceleration subproblems;
- `prox_l1_l1.m` : function evaluating the proximal operator associated with the fused-lasso regularization.

See the documentation inside each file for further details.

## Example of use
- `sbsa_qp_demo.m`     : example of use of SBSA_QP on the FF48-30y problem considered in [1];
- `ff48_data_montly.m` : function loading the FF48 data;
- `EvaluateNaive.m`    : function evaluating the naive portfolio;
- `EvaluateRisk.m`     : function evaluating the risk of a given portfolio.

## License
[![GNU GPL v3.0](http://www.gnu.org/graphics/gplv3-127x51.png)](http://www.gnu.org/licenses/gpl.html)

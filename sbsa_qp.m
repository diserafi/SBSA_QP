% -------------------------------------------------------------------------
% Copyright (C) 2020 by V. De Simone, D. di Serafino, M. Viola.
%
%                           COPYRIGHT NOTIFICATION
%
% This file is part of SBSA.
%
% SBSA is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% SBSA is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with SBSA. If not, see <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------
       
function [u,fopt,iter,errA,outargs] = sbsa_qp(Q,p,isLS,tau1,D,tau2,A,b,options)

%==========================================================================
%
% AUTHORS:
%   Valentina De Simone (valentina[dot]desimone[at]unicampania[dot]it)
%   Daniela di Serafino (daniela[dot]diserafino[at]unicampania[dot]it)
%   Marco Viola         (marco[dot]viola       [at]unicampania[dot]it)
%
% VERSION: 1.0
% Last Update: 28 May 2020
%
%==========================================================================
%
% This function implements the "Split Bregman method with Subspace 
% Acceleration" (SBSA), for the solution of constrained optimization
% problems of the form
%
% (1)           min  f(u) + tau_1 ||u||_1 + tau_2*||D*u||_1
%               s.t. A*u = b,
%
% modeling, e.g., multiperiod portfolio optimization problems, regularized
% least-squares regression problems or source detection problems in
% electroencephalography.
%
% Here f(u) is a quadratic function, having either the form
%
% (2)           f(u) = 0.5*u'*Q*u - p'*u,
%
% or the least squares form
%
% (3)           f(u) = 0.5*||Q*u - p||^2.
%
% SBSA is an iterative method based on the Split Bregman method
% [T. Goldstein and S. Osher, SIAM J. Imaging Sciences, 2 (2009), pp.323-343]
% with the introduction of subspace-acceleration steps.
% The subspaces where the acceleration is performed are selected so that 
% the restriction of the objective function is a smooth function in a 
% neighborhood of the current iterate. For simplicity we will refer to the
% non-accelerated steps as "standard Bregman steps".
%
% The minimization in standard Bregman steps is performed by using the FISTA
% method implemented in the FOM package [A. Beck and N. Guttmann-Beck,
% Optimization Methods and Software, 34:1 (2019), pp. 172-193]. 
% The minimization of the unconstrained quadratic problems in the accelerated
% steps is performed by using the CG method.
%
% The algorithm stops if the violation of the linear constraints is below a
% given threshold, namely
%
%              ||A*u-b|| <= tol_B    and    ||D*u-d|| <= tol_B,
%
% where d is the auxiliary variable introduced in the Bregman scheme (see
% Section 3 in [1] for further details).
%
% =========================================================================
%
% SOFTWARE REQUIREMENTS:
% SBSA_QP requires the FOM package(https://sites.google.com/site/fomsolver/);
% note that the path to FOM must be added to the MATLAB path with the command
%      addpath(genpath('path_to_fom'))
%
%==========================================================================
%
% REFERENCES:
% [1] V. De Simone, D. di Serafino and M. Viola,
%     "A subspace-accelerated split Bregman method for sparse data recovery
%      with joint l_1-type regularizers", Electronic Transactions on
%      Numerical Analysis (ETNA), volume 53, 2020, pp. 406-425,
%      DOI: 10.1553/etna_vol53s406.
%
% Preprint available from ArXiv
%     https://arxiv.org/abs/1912.06805
% and Optimization Online
%     http://www.optimization-online.org/DB_HTML/2019/12/7519.html
% 
%==========================================================================
%
% INPUT ARGUMENTS:
%
% Q       = double matrix, representing either the Hessian of the function
%           f(u) in (2) (in this case it must be a square matrix) or the
%           matrix in the LS term in (3);
% p       = double array, representing either the linear term of the
%           function f(u) in (2) or the r.h.s. of the LS term in (3);
% isLS    = logical, true if f(u) is defined as in (3);
% tau1    = double, weight of the first L1-regularization term in (1);
% D       = double matrix, defines the second L1-regularization term
%           in (1);
% tau2    = double, weight of the second L1-regularization term in (1);
% A       = double matrix, coefficients of the linear constraints;
% b       = double array, r.h.s. of the linear constraints;
% options = [optional] struct array with the following (possible) entries,
%           to be specified as pairs ('propertyname', propertyvalue);
%           the string 'propertyname' can be:
%           tol_B   = double, tolerance on the violation of the linear
%                     constraints [default 1e-4];
%           maxit   = integer, maximum number of Bregman iterations 
%                     [default 5000];
%           tol_F   = double, tolerance for the stopping criterion of
%                     FISTA in the standard Bregman steps [default 1e-5];
%           tol_CG  = double, relative tolerance on the residual norm for
%                     the termination of the CG method in the accelerated
%                     steps [default 1e-2]; 
%           gamma   = double, starting value for the parameter used to
%                     switch between standard and accelerated steps (see
%                     Sections 4 and 5.2 of [1] for details) [default 10];
%           verbose = integer, values 0, 1, 2 
%                      0: no print [default],
%                      1: print information at final step,
%                      2: print information at each step.
% 
% OUTPUT ARGUMENTS:
%
% u       = double array, computed solution;
% fopt    = double, objective function value;
% iter    = integer, total number of Bregman iterations performed;
% errA    = double, violation of the linear constraints A*u = b;
% outargs = struct array, containing further information on the execution,
%           i.e.:
%           errD    = double, violation of the linear constraints D*u - d = 0;
%           it_std  = integer, total number of standard iterations performed;
%           tot_fista_steps = integer, total number of FISTA steps performed;
%           it_subacc = integer, total number of subspace-accelerated
%                       iterations performed;
%           tot_cg_steps = integer, total number of CG steps performed;
%           total_time = elapsed time.
%
%==========================================================================

%% Initializing external parameters (can be passed to SBSA_QP with the "options" structure)
tol_B   = 1e-4;
maxit   = 5000;
tol_F   = 1e-5;
tol_CG  = 1e-2;
gamma   = 10;
verbose = 0;

%% Grabbing personalized settings from options
optionnames = fieldnames(options);
for iter=1:numel(optionnames)
    switch upper(optionnames{iter})
        case 'TOL_B'
            tol_B = options.(optionnames{iter});
        case 'MAXIT'
            maxit = options.(optionnames{iter});           
        case 'TOL_F'
            tol_F = options.(optionnames{iter});
        case 'TOL_CG'
            tol_CG = options.(optionnames{iter});
        case 'GAMMA'
            gamma = options.(optionnames{iter});
        case 'VERBOSE'
            verbose = options.(optionnames{iter});
        otherwise
            error(['Unrecognized option: ''' optionnames{iter} '''']);
    end
end

%% Starting the clock
tstart = tic;

%% Initializing other parameters
warmstart = 5;      % integer, number of standard Bregman iteration to perform before allowing acceleration
vargamma = true;    % logical, adapt the value of gamma (see Section 5.2 in [1])
eps_act = 0;        % double, threshold for "activity"
fracstepCG = 0.5;   % double, the maximum number od CG steps is set to fracstepCG*subproblem_size;
lam1 = 1;           % double, weight of the penalty term ||A*u - bkA||^2
lam2 = 1;           % double, weight of the penalty term ||D*u - d - bkD||^2
etaLS = 1e-1;       % double, coefficient for the sufficient decrease in the projected linesearch

%% Setting FISTA parameters
par_fista = struct();
par_fista.max_iter    = 5000;
par_fista.print_flag  = false;
par_fista.Lstart      = 1;
par_fista.regret_flag = true;
par_fista.eps         = tol_F;

%% Setting starting point and initializing auxiliary vectors
[n_d,n_u] = size(D);
n_b = length(b);
u   = zeros(n_u,1);
d   = zeros(n_d,1);
x   = [u;d];
bAk = zeros(n_b,1);
bDk = zeros(n_d,1);
if isLS
    QT = Q';
end
AT = A';
DT = D';
tauvec = [tau1*ones(n_u,1);tau2*ones(n_d,1)];

%% Other initializations
errA = 1;
errD = 1;
timetoleave = (errA <= tol_B && errD <= tol_B);
errALred = 0;
last_is_std = 1;
it_std = 0;
tot_fista_steps = 0;
it_subacc = 0;
tot_cg_steps = 0;
iter = 0;

%% Main loop starts

while (~timetoleave && iter<maxit) || (timetoleave && last_is_std)

    iter = iter+1;
    
    % Updating Bregman vector associated with constraint Au = b
    Au = A*u;
    bAk = bAk+(b-Au);

    % Updating Bregman vector associated with constraint Du - d = b
    bDk = bDk + D*u - d;

    % Updating quadratic function objects
    if isLS
        func = @(x)0.5*norm(Q*x(1:n_u)-p,2)^2+lam1*0.5*norm(A*x(1:n_u)-bAk,2)^2+lam2*0.5*norm(x(n_u+1:end)-D*x(1:n_u)-bDk,2)^2;
        grad = @(x)[QT*(Q*x(1:n_u)-p)+lam1*AT*(A*x(1:n_u)-bAk)-lam2*DT*(x(n_u+1:end)-D*x(1:n_u)-bDk); lam2*(x(n_u+1:end)-D*x(1:n_u)-bDk)];
        hessprod = @(x)[QT*(Q*x(1:n_u))+lam1*AT*(A*x(1:n_u))-lam2*DT*(x(n_u+1:end)-D*x(1:n_u)); lam2*(x(n_u+1:end)-D*x(1:n_u))];
    else
        func = @(x)0.5*x(1:n_u)'*Q*x(1:n_u) - x(1:n_u)'*p +lam1*0.5*norm(A*x(1:n_u)-bAk,2)^2+lam2*0.5*norm(x(n_u+1:end)-D*x(1:n_u)-bDk,2)^2;
        grad = @(x)[Q*x(1:n_u) - p + lam1*AT*(A*x(1:n_u)-bAk)-lam2*DT*(x(n_u+1:end)-D*x(1:n_u)-bDk); lam2*(x(n_u+1:end)-D*x(1:n_u)-bDk)];
        hessprod = @(x)[Q*x(1:n_u) + lam1*AT*(A*x(1:n_u))-lam2*DT*(x(n_u+1:end)-D*x(1:n_u)); lam2*(x(n_u+1:end)-D*x(1:n_u))];
    end
    reg  = @(x)tau1*norm(x(1:n_u),1)+tau2*norm(x(n_u+1:end),1);
    
    % Checking activity
    Sz = abs(x) <= eps_act;
    Sp = (x >  eps_act);  
    Sn = (x < -eps_act);  

    % Computing gradient, phi and beta
    gradf    = grad(x);
    beta     = setBeta(x,gradf,tauvec,Sz);
    normbeta = norm(beta);
    phi      = setPhi(x,gradf,tauvec,Sp,Sn);
    normphi  = norm(phi);

    moving = (Sz==0);
  
    if ( ((normbeta <= gamma*normphi) && errALred) || (timetoleave && last_is_std) ) && iter > warmstart
    %% Performing subspace-acceleration step
        last_is_std = 0;
        it_subacc = it_subacc + 1;

        % Updating gamma
        if vargamma
            gamma = max(1,gamma*0.9);
        end

        % Setting Hessian and r.h.s. for CG
        Hv      = @(v) hessprod(v);
        SandSp  = (moving+Sp == 2);
        SandSn  = (moving+Sn == 2);
        rhs     = gradf + tauvec.*SandSp - tauvec.*SandSn;
        ind     = ~moving;

        % Subproblem solution
        maxitCG = floor(fracstepCG*nnz(moving));
        [dir, HvProds, ~ ] = CGsubspace (Hv , rhs, ind, tol_CG, maxitCG);

        % Projected linesearch        
        alpha = 1;
        y = x+dir;        
        y = project(y,SandSp,SandSn);
        DirDer = sum( rhs(moving).*dir(moving) );
        
        F_old = func(x) ;
        absx  = abs(x);
        normU = sum(absx(1:n_u));       % 1-norm of vector u
        normD = sum(absx(n_u+1:end));   % 1-norm of vector d
        F_old = F_old + tau1*normU + tau2*normD;
        
        f       = func(y);
        absy    = abs(y);
        normU_y = sum(absy(1:n_u));
        normD_y = sum(absy(n_u+1:end));
        F = f + tau1*normU_y + tau2*normD_y;
        
        while(1)
            if F - F_old <= etaLS*alpha*DirDer
                break
            end
            if alpha < 1e-12
               break
            end
            
            alpha   = alpha*0.5;
            y       = x + alpha*dir;
            y       = project(y,SandSp,SandSn);
            f       = func(y);
            absy    = abs(y);
            normU_y = sum(absy(1:n_u));
            normD_y = sum(absy(n_u+1:end));
            F       = f + tau1*normU_y + tau2*normD_y;
        end
        
        x = y;

        u = x(1:n_u);
        d = x(n_u+1:end);
        
        % Computing violation of linear constraints
        errAold = errA;
        vv = A*u-b;
        errA = norm(vv);
        errLold = errD;
        dxd = d-D*u;
        errD = norm(dxd);
        errALred = (errA <= errAold && errD <= errLold);

        tot_cg_steps = tot_cg_steps + HvProds;
        
        if verbose > 1
            if isLS
                fopt = 0.5*norm(Q*u-p,2)^2 + tau1*norm(u,1) + tau2*norm(D*u,1);
            else
                fopt = 0.5*u'*Q*u - p'*u   + tau1*norm(u,1) + tau2*norm(D*u,1);
            end
            fprintf ('\niter = %4d -- acc -- nnz(u) = %4d --  cg_steps = %4d (%4d) --  tot_cg_steps = %7d -- funobj = %e -- ||Aw-b|| = %e -- ||Lw-d|| = %e',...
                iter,nnz(u),HvProds,maxitCG,tot_fista_steps,fopt,errA,errD);
        end
    
    else
    %% Performing standard Bregman step
        last_is_std = 1;
        it_std = it_std + 1;

    
        if vargamma && exist('HvProds','var')
            gamma = max(1,gamma*1.1);
        end
        
        % Solving subproblem with fista
        [x,~,parout] = fista(func,grad,reg,@(x,a)prox_l1_l1(x,n_u,tau1*a,tau2*a),1,x,par_fista);
        
        % Keeping track of the Lipschitz constant estimate
        par_fista.Lstart = parout.LValVec(end);
        
        u = x(1:n_u);
        d = x(n_u+1:end);
        
        % Computing violation of linear constraints
        errAold = errA;
        vv = A*u-b;
        errA = norm(vv);
        errLold = errD;
        dxd = d-D*u;
        errD = norm(dxd);
        errALred = (errA <= errAold && errD <= errLold);

        tot_fista_steps = tot_fista_steps + parout.iterNum;
        
        if verbose > 1
            if isLS
                fopt = 0.5*norm(Q*u-p,2)^2 + tau1*norm(u,1) + tau2*norm(D*u,1);
            else
                fopt = 0.5*u'*Q*u - p'*u   + tau1*norm(u,1) + tau2*norm(D*u,1);
            end
            fprintf ('\niter = %4d -- std -- nnz(u) = %4d -- fom_steps = %6d      -- tot_fom_steps = %7d -- funobj = %e -- ||Aw-b|| = %e -- ||Lw-d|| = %e',...
            iter,nnz(x),parout.iterNum,tot_fista_steps,fopt,errA,errD);
        end

    end

    % Checking stopping criterion
    if ~timetoleave
        timetoleave = (errA <= tol_B && errD <= tol_B);
    end

end

total_time = toc(tstart);

%% Printing information at final step
if isLS
    fopt = 0.5*norm(Q*u-p,2)^2 + tau1*norm(u,1) + tau2*norm(D*u,1);
else
    fopt = 0.5*u'*Q*u - p'*u   + tau1*norm(u,1) + tau2*norm(D*u,1);
end
if verbose > 0
    fprintf ('\nSBSA execution terminated');
    fprintf ('\niter = %4d -- nnz(u) = %4d -- Fopt = %e -- ||Aw-b|| = %e -- ||Lw-d|| = %e\n', ...
        iter,nnz(u),fopt,errA,errD);
end

outargs = struct('errD',errD,'it_std',it_std,'tot_fista_steps',tot_fista_steps,...
                'it_subacc',it_subacc,'tot_cg_steps',tot_cg_steps,'total_time',total_time);

end


%%
%==========================================================================
%                            AUXILIARY FUNCTIONS
%==========================================================================

%% Set beta
function [beta,index_1,index_2] = setBeta(X,grad,tauvec,Sz)
    beta = zeros(size(X));    
    index_1 = (grad < -tauvec & Sz==1);
    index_2 = (grad >  tauvec & Sz==1);
    beta(index_1) = grad(index_1)+tauvec(index_1);
    beta(index_2) = grad(index_2)-tauvec(index_2);
end

%% Set phi
function phi = setPhi(X,gradf,tauvec,Sp,Sn)
    zer = zeros(size(X));
    phi = zeros(size(X));
    index_1 = (Sp==1 & gradf > -tauvec);
    index_2 = (Sn==1 & gradf <  tauvec);
    phi(Sp) = gradf(Sp) + tauvec(Sp);
    phi(Sn) = gradf(Sn) - tauvec(Sn);
    phi(index_1) = min( [ phi(index_1), max([zer(index_1),X(index_1),gradf(index_1)-tauvec(index_1)],[],2)], [], 2 ); 
    phi(index_2) = max( [ phi(index_2), min([zer(index_2),X(index_2),gradf(index_2)+tauvec(index_2)],[],2)], [], 2 );
end

%% Project y onto the orthant containing x
function proj_y = project(y,Sp,Sn)
    proj_y     = y;
    proj_y(Sp) = max(y(Sp),0);
    proj_y(Sn) = min(y(Sn),0);
end

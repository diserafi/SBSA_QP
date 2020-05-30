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
%
% Authors:
%   Valentina De Simone (valentina[dot]desimone[at]unicampania[dot]it)
%   Daniela di Serafino (daniela[dot]diserafino[at]unicampania[dot]it)
%   Marco Viola         (marco[dot]viola       [at]unicampania[dot]it)
%
% Version: 1.0
% Last Update: 28 May 2020
%
%==========================================================================
% 
% DESCRIPTION:
% This is a demo showing how to use "sbsa_qp" to solve a sparse portfolio
% optimization problem with fused-lasso regularization.
% The script loads the FF48 data, builds up the matrices characterizing
% the problem, solves it by calling "sbsa_qp" and then evaluates performance
% metrics [2].
% See Section 5 in [1] for further details.
% 
%==========================================================================
%
% REFERENCE PAPER:
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
% 
% OTHER REFERENCES:
% [2] S. Corsaro, V. De Simone and Z. Marino,
%     "Fused Lasso approach in portfolio selection", Annals of Operations
%      Research, 2019, DOI: 10.1007/s10479-019-03289-w.
% 
%==========================================================================

%% NOTE: add FOM solver to the MATLAB path

%% Loading and processing the FF48 monthly averaged data
TEST='FF48';
[Rtot] = ff48_data_montly();
[~,M] = size(Rtot);
M = M-1;
% NOTE: We are considering the 30 year period
% July 1985 - June 2015
N = 29;
N1 = N;

% Constraint on initial budget
xinit = 1.0;

for i = 1:N  % loop over periods
    R = Rtot(589+12*(i-1):708+12*(i-1),2:M+1)./100;
    C(:,:,i) = cov(R);
    % mean return for each asset
    meanreturn1 = mean(R,1);
    % mean return for each asset over annual basis
    rbar(i,:) = (meanreturn1*12);
end

dim = N*M;
S = sparse(dim,dim);
istart = 1;
for i = 1:N
    chol(C(:,:,i)); % check positive definiteness 
    S(istart:istart+M-1,istart:istart+M-1) = C(:,:,i);
    istart = istart+M;
end

%% Building naive portfolio
[ueq,xterm] = EvaluateNaive(xinit,M,N,rbar); % naive portfolio
soleqT = reshape(ueq,M,N+1);
soleq = soleqT'; % size: (N+1,M)

% Evaluate naive portfolio metrics
solgeq = soleq;
Ng = N+1;
for i=2:Ng
    CTMeq(i-1,:) = solgeq(i,:)~=solgeq(i-1,:);
end
CTMeq = double(CTMeq);
xx = solgeq(1,:);
xx(find(xx)) = 1;
CTMeq=[xx; CTMeq];

% Transaction cost of naive portfolio
CTeq = sum(sum(CTMeq));
% Max number of transactions over assets (asset with max # transactions)
norm1eq = norm(CTMeq,1);
% Max number of transactions over periods (period with max # transactions)
norminfeq = norm(CTMeq,Inf);

%% Building linear operator for fused-lasso regularization
dim1 = (N-1)*M;
D = -speye(dim1,dim);
E = sparse(1:dim1,1+M:dim1+M,ones(1,dim1),dim1,dim);
D = D+E;
clear E;

%% Build linear constraints coefficients and rhs
b = zeros(N+1,1); 
b(1,:) = xinit;
b(N+1,:) = xterm;

A = zeros(N+1,dim);
A(1,1:M) = 1;
for i = 1:N-1
    istart = M*(i-1)+1;
    iend   = M*i;
    A(i+1, istart:iend) = 1+rbar(i,:); 
    istart = iend+1;
    iend   = istart+M-1;
    A(i+1, istart:iend) = -1;
end
A(end, end-M+1:end) = 1.0+rbar(end,:);
A = sparse(A);

%% Solving the problem using SBSA_QP
options = struct('verbose',1,...  Printing information on the termination
            'tol_B',1e-4,'tol_F',1e-5,'tol_CG',1e-2); % Tolerances
tau1 = 1e-2;
tau2 = 1e-2;
tic,
[u,fopt,iter,errA,outargs] = sbsa_qp(S,zeros(dim,1),0,tau1,D,tau2,A,b,options);
toc
%% Evaluating metrics for the computed solution
% Thresholding the solution
ufin = u;
thrs = 1e-4;

u(abs(u)<xinit*thrs) = 0;
solT = reshape(u,M,N);
sol = solT';

solg = [sol; (sol(end,:)) .*(1+rbar(end,:))];
for i = 1:Ng
    % normalized weight matrix - size (N+1,M)
    mom(i,:)   = solg(i,:)/sum(solg(i,:));
    % normalized naive weight matrix - size (N+1,M)
    momeq(i,:) = solgeq(i,:)/sum(solgeq(i,:));
end

% Number of short positions
n_short = length(find(mom<0));

% Differences among consecutive rows, i.e., consecutive periods
mom_diff = diff(solg);
mom_diff(abs(mom_diff)<xinit*thrs) = 0;

% Transaction costs
CT_mom_diff=length(find(mom_diff))+length(find(solg(1,:)~=0));
LogicMomDiff = mom_diff;
LogicMomDiff(find(LogicMomDiff)) = 1;
xx = solg(1,:);
xx(find(xx)) = 1;
LogicMomDiff = [xx; LogicMomDiff];
clear xx

% Max # transactions over assets (asset with max # transactions)
norm1 = norm(LogicMomDiff,1);
% Max # transactions over periods (period with max # transactions)
norminf = norm(LogicMomDiff,Inf);


%% Printing quality metrics for the two portfolios
% obj function - sum of variances over periods
fo   = EvaluateRisk(u,M,N,C);
foeq = EvaluateRisk(ueq,M,N,C);
fprintf('\nNaive portfolio risk = %e  |  Optimal portfolio risk = %e  |  Ratio = %e\n',foeq,fo,foeq/fo);
% density
active = length(find(mom));
active_eq = length(find(momeq));
density = active/(M*Ng);
density_eq= active_eq/(M*Ng);
fprintf('Naive portfolio:   density = %e   TC = %5d   1-norm = %5d   inf-norm = %5d\n',density_eq,CTeq,norm1eq,norminfeq);
fprintf('Optimal portfolio: density = %e   TC = %5d   1-norm = %5d   inf-norm = %5d\n',density,CT_mom_diff,norm1,norminf);
fprintf('                   short pos = %5d        sparsity (assets vs periods matrix) = %5d\n',n_short,nnz(mom));


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
       
function [ p, num_cg, iflag ] = CGsubspace ( Hv, b, ind, errtol, maxit)
%==========================================================================
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
% This function implements CG-Steihaug for the solution of linear systems
% of the form
%                        H*p = -b
% restricted to the subspace defined by the vector "ind" as follows:
%                        p_i = 0 if ind(i) = 1.
% As a safeguard, the problem is also subject to the trust-region
%                        ||p||_2 < 10^3.
% This implementation is a modification of the CG subspace solver which can
% be found in the OBA software by Nitish Shirish Keskar
% (https://github.com/keskarnitish/OBA).
%
%==========================================================================
%
% INPUT ARGUMENTS
%
% Hv     = function handle to the matrix-vector product H*v;
% b      = double array, -rhs of the linear system;
% ind    = logical array, array indices to be considered (if ind(i) = 1 
%          then the i-th entry of a given vector is set to 0);
% errtol = double, relative tolerance on the residual norm;
% maxit  = integer, maximum number of iterations.
% 
% OUTPUT ARGUMENTS
% 
% p      = double array, computed solution;
% num_cg = integer, number of iteration performed;
% iflag  = string, with possible values
%           'RS' : the relative tolerance on the residual was satisfied,
%           'MX' : the maximum number of iterations was reached,
%           'TR' : the boundary of the trust region was hit,
%           'NC' : the algorithm found a negative-curvature direction.
%
%==========================================================================

n     = length(b);
delta = 1e3;
iprnt  = 0;
iflag  = ' ';
%
g      = b;
g(ind) = 0;
x      = zeros(n,1);
r      = -g;
%
z    = r;
rho  = z'*r;
tst  = norm(r,'inf');
flag = '';
terminate = errtol*norm(g,'inf');   it = 1;    hatdel = delta*(1-1.d-6);
rhoold = 1.0d0;
if iprnt > 0
    fprintf(1,'\n\tThis is an output of the CG-Steihaug method. \n\tDelta = %7.1e \n', delta);
    fprintf(1,'   ---------------------------------------------\n');
end
flag = 'We do not know ';
if tst <= terminate; flag  = 'Small ||g||    '; end
w = x;

while((tst > terminate) && (it <=  maxit) && norm(x) <=  hatdel)
    %
    if(it == 1)
        p = z;
    else
        beta = rho/rhoold;
        p = z + beta*p;
    end
    %
    %
    p(ind) = 0;
    w  = Hv(p);
    w(ind) = 0;
    
    alpha = w'*p;
    %
    % If alpha < = 0 head to the TR boundary and return
    %
    ineg = 0;
    if(alpha <=  0)
        ac = p'*p; bc = 2*(x'*p); cc = x'*x - delta*delta;
        alpha = (-bc + sqrt(bc*bc - 4*ac*cc))/(2*ac);
        flag  = 'negative curvature';
        iflag = 'NC';
    else
        alpha = rho/alpha;
        if norm(x+alpha*p) > delta
            ac = p'*p; bc = 2*(x'*p); cc = x'*x - delta*delta;
            alpha = (-bc + sqrt(bc*bc - 4*ac*cc))/(2*ac);
            flag  = 'boundary was hit';
            iflag = 'TR';
        end
    end
    x   =  x + alpha*p;
    r   =  r - alpha*w;
    tst = norm(r,'inf');
    if tst <= terminate; flag = '||r|| < test   '; iflag = 'RS'; end;
    if norm(x) >=  hatdel; flag = 'close to the boundary'; iflag = 'TR'; end
    
    if iprnt > 0
        fprintf(1,' %3i    %14.8e   %s  \n', it, tst, flag);
    end
    rhoold = rho;
    z   = r;
    rho = z'*r;
    it  = it + 1;
end %

if it > maxit; iflag = 'MX'; end;

num_cg = it-1;
p = x;

end
%
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

function out = prox_l1_l1(x,n_1,alpha1,alpha2)

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
% This function computes the proximal operator of the function 
%         alpha1*norm(x(1:n_1),1) + alpha_2*norm(x(n_1+1:end),1)
%
%==========================================================================
%
% INPUT ARGUMENTS:
%
%  x      = double array, point in which to evaluate the proximal operator;
%  n_1    = integer, size of the firse part of the vector, subject to the
%           regularization term alpha1*norm(x(1:n_1),1);
%  alpha1 = double, weight of the first regularization term;
%  alpha2 = double, weight of the second regularization term.
% 
% OUTPUT ARGUMENTS:
%
%  out    = double, proximal operator at x.
% 
%==========================================================================
% 
% INFO ON THE ORIGINAL VERSION:
% 
% This file is a modification of PROX_L1, which is part of the FOM package -
% a collection of first order methods for solving convex optimization problems 
% Copyright (C) 2017 Amir and Nili Beck
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%==========================================================================

if (nargin < 4)
    alpha2 = alpha1 ;
end

if (alpha1 < 0) || (alpha2 < 0)
    error('alpha1 and alpha2 should be positive')
end

absX = abs(x);
signX = sign(x);
N = length(x);
alpha = [ones(n_1,1)*alpha1;ones(N-n_1,1)*alpha2];
out = max(absX - alpha,0).* signX;
end


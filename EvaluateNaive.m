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

function [ueq,xterm] = EvaluateNaive(xinit,M,N,rbar)
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
% This function evaluates the naive portfolio.
% 
%==========================================================================

dim = (N+1)*M;
ueq = zeros(dim,1);
wealth = zeros(N+1,1);
weq = xinit*ones(1,M)./M;
wealth(1) = xinit;
ueq(1:M) = weq;

for i=2:N
    val = ueq(M*(i-2)+1:(i-1)*M)' .* (1.0+ rbar(i-1,:)); 
    wealth(i) = sum(val); 
    ueq(M*(i-1)+1:i*M) = ( wealth(i)*ones(1,M) )./M;
end

ueq(end-M+1:end) = ueq(end-2*M+1:end-M)' .*(1+rbar(end,:));
xterm = sum(ueq(end-M+1:end));
ueq(end-M+1:end) = (xterm*ones(1,M))./M;

end


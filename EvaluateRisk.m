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

function risk = EvaluateRisk(y,M,N,C)
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
% This function computes the risk of a given portfolio.
% 
%==========================================================================

risk = 0;
for i = 1:N
    init = M*(i-1)+1;
    fine = init+M-1;
    vet = y(init:fine); % column vector
    Ci = C(:,:,i)*vet;  % C_i* u_i    
    risk = risk + 0.5*vet'*Ci; 
    clear Ci; clear vet; 
end

end
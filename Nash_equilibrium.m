% A demo for finding the mixed-strategy Nash equilibrium
% The cvx toolbox is required.
% If you use our data, a NE point {P_fa1=0.04 , r='PGD02'} should be
% obtained/

clear, clc

PDname = 'PD_Pfa05.mat';
load(PDname);
cvx_begin quiet
    variables v y(size(PD, 1))
    minimize v
    subject to
    y>=0;
    sum(y)==1;
    PD'*y<=v;
cvx_end
support = find(y>0.001);
Mixed_NEvalue= cvx_optval
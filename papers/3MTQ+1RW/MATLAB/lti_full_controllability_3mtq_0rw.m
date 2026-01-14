clear; clc;
tf = 600;       
J_cage = diag([1; 1.2; 0.9]); 

% Reaction Wheels
A_rw = [];
D_rw = [];
h_0 = [];
n = 6 + size(A_rw, 2);

% Magnetorquers
A_mtq = [1, 0, 0; % MTQ 1
        0, 1, 0; % MTQ 2
        0, 0, 1]';

% B-Field in Inertial Frame
B_i = [1; 1; 1]*1e-6;

%% Building Linearized Matrices
[s, q, r] = lti_rank(J_cage, A_rw, D_rw, h_0, A_mtq, B_i);
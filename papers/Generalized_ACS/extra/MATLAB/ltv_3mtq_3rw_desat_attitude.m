clear; clc;
tf = 600;       
J_cage = diag([1; 1.2; 0.9]); 

% Reaction Wheels
A_rw = [1, 0, 0; % RW 1
        0, 1, 0; % RW 2
        0, 0, 1;]'; % RW 3 
D_rw = diag([0.1; 0.1; 0.1]); 
h_0 = [0; 0; 0];
n = 6 + size(A_rw, 2);

% Magnetorquers
A_mtq = [1, 0, 0; % MTQ 1
        0, 1, 0; % MTQ 2
        0, 0, 1]';

% Magnetic Field
VARYING_B_FIELD = true;
B0_phys = 1e-6; 
if VARYING_B_FIELD
    period = 5400;
    B_fun = @(t) B0_phys * [cos(2*pi*t/period); sin(2*pi*t/period); 0.5*cos(2*pi*t/(period/2))];
else
    dir = [1; 0.5; 0.2];
    B_fun = @(t) B0_phys * (dir / norm(dir));
end

[r, n] = ltv_rank(J_cage, A_rw, D_rw, h_0, A_mtq, B_fun, tf, true, "full");
fprintf('LTI task-controllability rank = %d / %d\n', r, n);
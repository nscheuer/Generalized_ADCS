%% Setup
% Satellite Properties
J_noRW = diag([1; 1; 1]);

% RW Properties
N_rw = 3;
A_rw = [1, 0, 0; % RW 1
    0, 1, 0; % RW 2
    0, 0, 1;]'; % RW 3
D_rw = diag([0.1; 0.1; 0.1]);

% MTQ Properties
N_mtq = 3;
A_mtq = [1, 0, 0; % MTQ 1
    0, 1, 0; % MTQ 2
    0, 0, 1]'; % MTQ 3

% Linearization Properties
B_i = [0; 0; 1]*1e-6; % Inertial magnetic field in Tesla
h_0 = [0; 0; 0]; % Equilibrium RW momentum

%% Building Linearized Matrices
A = [zeros(3,3), 1/4*eye(3);
    zeros(3,3), inv(J_noRW)*crossmat(A_rw*h_0)];

B = [zeros(3,N_rw), zeros(3,N_mtq);
    inv(J_noRW)*A_rw, -inv(J_noRW)*crossmat(B_i)*A_mtq];

%% Check Controllability
Co = ctrb(A,B);
unco = length(A) - rank(Co)
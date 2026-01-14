%% Setup
clear;clc;
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
B_i = [1; 1; 1]*1e-6; % Inertial magnetic field in Tesla
h_0 = [0; 0; 0]; % Equilibrium RW momentum

%% Building Linearized Matrices
A = [zeros(3,3), 1/4*eye(3), zeros(3, N_rw);
    zeros(3,3), inv(J_noRW)*crossmat(A_rw*h_0), zeros(3, N_rw);
    zeros(N_rw, 3), -D_rw*A_rw'*inv(J_noRW)*crossmat(A_rw*h_0), zeros(N_rw, N_rw)];

B = [zeros(3,N_rw), zeros(3,N_mtq);
    inv(J_noRW)*A_rw, -inv(J_noRW)*crossmat(B_i)*A_mtq;
    -eye(N_rw)-D_rw*A_rw'*inv(J_noRW)*A_rw, D_rw*A_rw'*inv(J_noRW)*crossmat(B_i)*A_mtq];

%% ================= Controllability Diagnostics =================

n = size(A,1);
m = size(B,2);

fprintf('\n=== System dimensions ===\n');
fprintf('States: %d (q=3, w=3, h_rw=%d)\n', n, N_rw);
fprintf('Inputs: %d (RW=%d, MTQ=%d)\n', m, N_rw, N_mtq);

%% Split B into RW and MTQ parts
B_rw  = B(:, 1:N_rw);
B_mtq = B(:, N_rw+1:end);

%% Controllability matrices
Co_all = ctrb(A, B);
Co_rw  = ctrb(A, B_rw);
Co_mtq = ctrb(A, B_mtq);

%% Singular values (full controllability matrix)
sv_all = svd(Co_all);

fprintf('\n=== Singular values of ctrb(A,B) ===\n');
format short e
disp(sv_all.');

%% Rank under different tolerances
tol_list = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14];

fprintf('\n=== Rank vs tolerance (ctrb(A,B)) ===\n');
for k = 1:length(tol_list)
    tol = tol_list(k);
    r = sum(sv_all > tol*max(sv_all));
    fprintf('  tol = %-8.1e  -> rank = %d / %d\n', tol, r, n);
end

%% RW-only controllability
sv_rw = svd(Co_rw);

fprintf('\n=== RW-only controllability ===\n');
disp(sv_rw.');
r_rw = sum(sv_rw > 1e-12*max(sv_rw));
fprintf('rank(ctrb(A,B_rw)) ≈ %d / %d\n', r_rw, n);

%% MTQ-only controllability
sv_mtq = svd(Co_mtq);

fprintf('\n=== MTQ-only controllability ===\n');
disp(sv_mtq.');
r_mtq = sum(sv_mtq > 1e-12*max(sv_mtq));
fprintf('rank(ctrb(A,B_mtq)) ≈ %d / %d\n', r_mtq, n);

%% Nullspace of controllability matrix (uncontrollable modes)
N_co = null(Co_all.', 'r');   % uncontrollable state directions

fprintf('\n=== Uncontrollable subspace ===\n');
fprintf('Dimension of uncontrollable subspace: %d\n', size(N_co,2));

if ~isempty(N_co)
    disp('Basis vectors (columns):');
    disp(N_co);
end

%% Project nullspace onto physical states
fprintf('\n=== Interpretation of uncontrollable modes ===\n');
if ~isempty(N_co)
    q_part = N_co(1:3, :);
    w_part = N_co(4:6, :);
    h_part = N_co(7:end, :);

    fprintf('Norms of components [q; w; h_rw]:\n');
    disp([vecnorm(q_part); vecnorm(w_part); vecnorm(h_part)]);
end

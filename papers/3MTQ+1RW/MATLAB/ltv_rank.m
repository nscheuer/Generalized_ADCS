function [s, r] = ltv_rank(J, A_rw, D_rw, h_0, A_mtq, B_fun, tf)
% LTV_RANK Computes singular values of the Controllability Gramian.
% Inputs:
%   J     : (3x3) Inertia matrix
%   A_rw  : (3xN_rw) RW axis matrix (can be [])
%   D_rw  : (N_rw x N_rw) RW inertia matrix (can be [])
%   h_0   : (N_rw x 1) Nominal wheel momentum (can be [])
%   A_mtq : (3xN_mtq) MTQ axis matrix (can be [])
%   B_fun : Function handle @(t) returning (3x1) B-field
%   tf    : Final integration time

    N_rw = size(A_rw, 2);
    N_mtq = size(A_mtq, 2);
    n = 6 + N_rw;

    if N_rw == 0
        A_rw = zeros(3, 0); D_rw = []; h_0 = [];
        H_rw_body = [0;0;0];
    else
        H_rw_body = A_rw * h_0;
    end
    
    if N_mtq == 0
        A_mtq = zeros(3, 0);
    end

    % Scaling for numerical stability
    S_rw = 1.0; 
    S_mtq = 1e6; 
    Scale_Matrix = diag([repmat(S_rw, 1, N_rw), repmat(S_mtq, 1, N_mtq)]);

    %% Build LTV Matrices
    A_core = [zeros(3,3), 0.25*eye(3);
              zeros(3,3), J \ crossmat(H_rw_body)];
    
    A_top_right = zeros(6, N_rw);
    A_bot_left  = [zeros(N_rw, 3), -D_rw*A_rw'*inv(J)*crossmat(H_rw_body)];
    A_bot_right = zeros(N_rw, N_rw);
    
    A_lin = [A_core, A_top_right; 
             A_bot_left, A_bot_right];

    B_phys = @(t) [ ...
        zeros(3, N_rw), zeros(3, N_mtq); ...
        J\A_rw, -J\crossmat(B_fun(t))*A_mtq; ...
        -eye(N_rw) - D_rw*A_rw'*inv(J)*A_rw,  D_rw*A_rw'*inv(J)*crossmat(B_fun(t))*A_mtq ...
    ];

    B_scaled = @(t) B_phys(t) * Scale_Matrix;

    %% Integration (Gramian)
    z0 = [reshape(eye(n),[],1); zeros(n^2,1)];
    opts = odeset('RelTol', 1e-12, 'AbsTol', 1e-14);
    
    [~, z] = ode45(@(t,z) ode_gramian(t, z, n, A_lin, B_scaled), [0, tf], z0, opts);

    Wc = reshape(z(end, n^2+1:end), n, n);
    Wc = 0.5 * (Wc + Wc.');
    s = svd(Wc);
    r = rank(Wc, 1e-10);

    fprintf('\n=== Singular Values (Scaled) ===\n');
    for i=1:length(s)
        fprintf('SV %d: %10.4e\n', i, s(i));
    end
    fprintf('Computed Rank (Tol 1e-10): %d / %d\n', r, n);
end

function dz = ode_gramian(t, z, n, A, B_fun)
    Phi = reshape(z(1:n^2), n, n);
    B_val = B_fun(t);
    if isempty(B_val), BBt = zeros(n); else, BBt = B_val * B_val.'; end
    
    Phi_dot = A * Phi;
    dz = [Phi_dot(:); reshape(Phi * BBt * Phi.', [], 1)];
end

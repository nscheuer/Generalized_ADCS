function [s, q, r] = ltv_rank_q(J, A_rw, D_rw, h_0, A_mtq, B_fun, tf)
% LTV_RANK_Q Computes rank and uncontrollable mode q via Stacked Matrix.
% Uses the condition: B(t)' * exp(A'*(T-t)) * q = 0
%
% Returns:
%   s : Singular values (comparable to Gramian SVs)
%   q : The most uncontrollable unit vector (at time T)
%   r : Rank estimate

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

    S_rw = 1.0; 
    S_mtq = 1e6; 
    Scale_Matrix = diag([repmat(S_rw, 1, N_rw), repmat(S_mtq, 1, N_mtq)]);

    %% Construct LTV Matrices
    A_core = [zeros(3,3), 0.25*eye(3);
              zeros(3,3), J \ crossmat(H_rw_body)];
    
    A_top_right = zeros(6, N_rw);
    A_bot_left  = [zeros(N_rw, 3), -D_rw*A_rw'*inv(J)*crossmat(H_rw_body)];
    A_bot_right = zeros(N_rw, N_rw);
    
    A = [A_core, A_top_right; 
         A_bot_left, A_bot_right];

    B_phys = @(t) [ ...
        zeros(3, N_rw), zeros(3, N_mtq); ...
        J\A_rw, -J\crossmat(B_fun(t))*A_mtq; ...
        -eye(N_rw) - D_rw*A_rw'*inv(J)*A_rw,  D_rw*A_rw'*inv(J)*crossmat(B_fun(t))*A_mtq ...
    ];
    B_scaled = @(t) B_phys(t) * Scale_Matrix;

    %% 3. Stacked Matrix Construction
    % We discretize time and stack the rows of B(t)' * Phi(T,t)'
    num_steps = 500; 
    time_grid = linspace(0, tf, num_steps);
    dt = time_grid(2) - time_grid(1);
    
    % Pre-allocate approximate size (num_inputs x n)
    total_inputs = N_rw + N_mtq;
    if total_inputs == 0
        % Degenerate case: No actuators
        s = zeros(n, 1); q = ones(n, 1); q = q/norm(q); r = 0;
        return;
    end
    
    O_stack = zeros(num_steps * total_inputs, n);
    
    % Loop to fill stack
    for k = 1:num_steps
        t = time_grid(k);
        
        % 1. State Transition Matrix Phi(T, t) = exp(A*(T-t))
        % Optimization: Since A is LTI, Phi(T,t) depends only on (T-t)
        Phi_T_t = expm(A * (tf - t));
        
        % 2. B matrix at time t
        Bt = B_scaled(t);
        
        % 3. Component: B(t)' * Phi(T, t)'
        % This maps the effect of input at time t to the state at time T
        row_block = Bt' * Phi_T_t';
        
        % 4. Fill Matrix
        idx_start = (k-1)*total_inputs + 1;
        idx_end   = k*total_inputs;
        O_stack(idx_start:idx_end, :) = row_block;
    end

    %% SVD
    [U, S_mat, V] = svd(O_stack, 'econ');
    s_raw = diag(S_mat);
    
    s = s_raw.^2 * dt;

    % Rank
    tol = max(size(O_stack)) * eps(max(s_raw));
    for i=1:length(s_raw)
        fprintf('SV %d: %10.4e\n', i, s_raw(i));
    end
    r = sum(s_raw > tol);

    % The Null Vector q (Uncontrollable Direction)
    % Corresponds to the smallest singular value (last column of V)
    q = V(:, end);

    %% 5. Reporting
    fprintf('\n=== LTV_RANK_Q Results ===\n');
    fprintf('Method: Stacked Reachability Matrix (samples=%d)\n', num_steps);
    fprintf('Computed Rank: %d / %d\n', r, n);
    fprintf('Smallest SV: %10.4e\n', min(s));
    
    if r < n
        fprintf('System is UNCONTROLLABLE.\n');
        fprintf('Uncontrollable Vector q (at time T):\n');
        disp(q');
    end
end
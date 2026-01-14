function [s, q, r] = lti_rank(J, A_rw, D_rw, h_0, A_mtq, B_vector)
% LTI_RANK Computes rank and uncontrollable modes for LTI systems.
%
% Usage: 
%   [s, q, r] = lti_rank(J, A_rw, D_rw, h_0, A_mtq, B_vector)
%
% Inputs:
%   J        : (3x3) Inertia matrix
%   A_rw     : (3xN_rw) RW axis matrix (can be [])
%   D_rw     : (N_rw x N_rw) RW inertia matrix (can be [])
%   h_0      : (N_rw x 1) Nominal wheel momentum (can be [])
%   A_mtq    : (3xN_mtq) MTQ axis matrix (can be [])
%   B_vector : (3x1) Constant Inertial Magnetic Field Vector [Tesla]
    
    % --- 1. Dimensions and Robust Setup ---
    N_rw = size(A_rw, 2);
    N_mtq = size(A_mtq, 2);
    n = 6 + N_rw; % State dimension (3 att, 3 rate, N_rw momentum)
    
    % Handle empty RW case safely
    if N_rw == 0
        A_rw = zeros(3, 0); 
        D_rw = []; 
        h_0 = [];
        H_rw_body = [0;0;0];
    else
        H_rw_body = A_rw * h_0;
    end
    
    % Handle empty MTQ case safely
    if N_mtq == 0
        A_mtq = zeros(3, 0);
    end
    
    % --- 2. Construct LTI System Matrices ---
    % Helper for cross product matrix
    crossmat = @(v) [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
    
    % Construct A Matrix
    A_core = [zeros(3,3), 0.25*eye(3);
              zeros(3,3), J \ crossmat(H_rw_body)];
    
    A_top_right = zeros(6, N_rw);
    
    % If N_rw=0, these bottom blocks shrink to empty dimensions automatically
    A_bot_left  = [zeros(N_rw, 3), -D_rw*A_rw'*inv(J)*crossmat(H_rw_body)];
    A_bot_right = zeros(N_rw, N_rw);
    
    A_sys = [A_core, A_top_right; 
             A_bot_left, A_bot_right];
         
    % Construct B Matrix (System Input Matrix)
    % B_sys is (n x (N_rw + N_mtq))
    
    % Block 1: Reaction Wheels contribution
    if N_rw > 0
        B_rw_part = [zeros(3, N_rw);
                     J \ A_rw;
                     -eye(N_rw) - D_rw*A_rw'*inv(J)*A_rw];
    else
        B_rw_part = zeros(n, 0);
    end
    
    % Block 2: Magnetorquers contribution
    if N_mtq > 0
        B_mtq_part = [zeros(3, N_mtq);
                      -J \ (crossmat(B_vector) * A_mtq);
                      D_rw*A_rw'*inv(J)*crossmat(B_vector)*A_mtq]; 
                      % Note: The last row exists only if N_rw > 0.
                      % If N_rw=0, MATLAB handles this concatenation correctly 
                      % because D_rw is empty.
         if N_rw == 0
             % Explicit fix for N_rw=0 case where the bottom block rows shouldn't exist
             B_mtq_part = [zeros(3, N_mtq);
                           -J \ (crossmat(B_vector) * A_mtq)];
         end
    else
        B_mtq_part = zeros(n, 0);
    end
    
    B_sys = [B_rw_part, B_mtq_part];

    % --- 3. Scaling for Numerical Stability ---
    % We scale the columns of B to normalize the actuator impact
    S_rw = 1.0; 
    S_mtq = 1e6; % MTQs are weak compared to wheels, scale up to see rank
    Scale_Matrix = diag([repmat(S_rw, 1, N_rw), repmat(S_mtq, 1, N_mtq)]);
    
    B_scaled = B_sys * Scale_Matrix;

    % --- 4. Global Controllability Analysis ---
    Co = ctrb(A_sys, B_scaled);
    
    [U, S_mat, ~] = svd(Co); 
    s = diag(S_mat);
    
    tol = max(size(Co)) * eps(max(s));
    r = sum(s > tol);
    
    % The Uncontrollable Vector q (smallest SV direction)
    q = U(:, end);

    % --- 5. Sub-System Diagnostics ---
    % Slicing the scaled matrix for sub-checks
    B_rw_scaled  = B_scaled(:, 1:N_rw);
    B_mtq_scaled = B_scaled(:, N_rw+1:end);
    
    rank_rw = 0;
    if N_rw > 0
        s_rw = svd(ctrb(A_sys, B_rw_scaled));
        rank_rw = sum(s_rw > tol);
    end
    
    rank_mtq = 0;
    if N_mtq > 0
        s_mtq = svd(ctrb(A_sys, B_mtq_scaled));
        rank_mtq = sum(s_mtq > tol);
    end

    % --- 6. Reporting ---
    fprintf('\n=== LTI_RANK Diagnostics ===\n');
    fprintf('System Size: n=%d | Inputs: RW=%d, MTQ=%d\n', n, N_rw, N_mtq);
    fprintf('Computed Rank: %d / %d\n', r, n);
    fprintf('Smallest SV:   %10.4e\n', min(s));
    
    fprintf('RW Only Rank:  %d / %d\n', rank_rw, n);
    fprintf('MTQ Only Rank: %d / %d\n', rank_mtq, n);
    
    if r < n
        fprintf('\n[!] System is UNCONTROLLABLE.\n');
        fprintf('Most Uncontrollable Direction q:\n');
        
        % Interpret q
        labels = {'Attitude (q)', 'Rate (w)', 'Momentum (h)'};
        
        idx_lims = [1, 3; 4, 6; 7, n];
        
        for i = 1:3
            if idx_lims(i,1) <= n
                end_idx = min(idx_lims(i,2), n);
                sub_q = q(idx_lims(i,1):end_idx);
                fprintf('  %s norm: %10.4f  | vec: %s\n', ...
                    labels{i}, norm(sub_q), mat2str(sub_q', 2));
            end
        end
    else
        fprintf('\n[OK] System is Fully Controllable.\n');
        fprintf('Weakest Direction (Smallest SV):\n');
        disp(q');
    end
end
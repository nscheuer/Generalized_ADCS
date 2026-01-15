function [r, n] = lti_rank(J, A_rw, D_rw, h_0, A_mtq, B_vector, control_h, task_mode, a_b)
%LTI_RANK  Rank test for LTI controllability of a task-relevant subspace z = P x.
%   Returns:
%     r : rank of task controllability matrix P*ctrb(A,B)
%     n : dimension of task-relevant subspace z \in R^n
%
%   Inputs:
%     J         (3x3) inertia
%     A_rw      (3xN_rw) RW axes (may be [])
%     D_rw      (N_rw x N_rw) RW inertia matrix (may be [])
%     h_0       (N_rw x 1) nominal wheel momentum (may be [])
%     A_mtq     (3xN_mtq) MTQ axes (may be [])
%     B_vector  (3x1) constant magnetic field vector (frame consistent with your linearization)
%     control_h (logical) include wheel momentum in state of interest and dynamics
%     task_mode (char/string) 'full' | 'vector' | 'vector_damping'
%       - 'vector'         : alignment + transverse rate damping
%       - 'vector_damping' : alignment + full rate damping (all 3 components)
%     a_b       (3x1, optional) body-fixed unit axis for vector pointing (default [0;0;1])

    if nargin < 9 || isempty(a_b)
        a_b = [0;0;1];
    end
    if nargin < 8 || isempty(task_mode)
        task_mode = 'full';
    end
    if nargin < 7 || isempty(control_h)
        control_h = false;
    end

    % Dimensions / safe empties
    N_rw  = size(A_rw,  2);
    N_mtq = size(A_mtq, 2);

    if N_rw == 0
        A_rw = zeros(3,0);
        D_rw = zeros(0,0);
        h_0  = zeros(0,1);
        H_rw_body = zeros(3,1);
    else
        if isempty(D_rw), D_rw = eye(N_rw); end
        if isempty(h_0),  h_0  = zeros(N_rw,1); end
        H_rw_body = A_rw * h_0;
    end

    if N_mtq == 0
        A_mtq = zeros(3,0);
    end

    % Helpers
    crossmat = @(v) [   0   -v(3)  v(2);
                      v(3)    0   -v(1);
                     -v(2)  v(1)    0 ];

    % ----------------------------
    % Build LTI A, B for chosen state
    % ----------------------------
    if control_h
        % Full state: x = [sigma; omega; h]
        n_x = 6 + N_rw;

        A_core = [zeros(3,3), 0.25*eye(3);
                  zeros(3,3), J \ crossmat(H_rw_body)];

        A_top_right = zeros(6, N_rw);
        A_bot_left  = [zeros(N_rw,3), -D_rw*A_rw'*(J \ crossmat(H_rw_body))];
        A_bot_right = zeros(N_rw, N_rw);

        A_sys = [A_core, A_top_right;
                 A_bot_left, A_bot_right];

        % Inputs: u = [u_rw; u_mtq]
        if N_rw > 0
            B_rw = [zeros(3,N_rw);
                    J \ A_rw;
                    -eye(N_rw) - D_rw*A_rw'*(J \ A_rw)];
        else
            B_rw = zeros(n_x, 0);
        end

        if N_mtq > 0
            B_mtq = [zeros(3,N_mtq);
                     -(J \ (crossmat(B_vector) * A_mtq));
                     D_rw*A_rw'*(J \ (crossmat(B_vector) * A_mtq))];
        else
            B_mtq = zeros(n_x, 0);
        end

        B_sys = [B_rw, B_mtq];

    else
        % Reduced state: x = [sigma; omega]
        n_x = 6;

        A_sys = [zeros(3,3), 0.25*eye(3);
                 zeros(3,3), J \ crossmat(H_rw_body)];

        if N_rw > 0
            B_rw  = [zeros(3,N_rw);
                     J \ A_rw];
        else
            B_rw  = zeros(n_x, 0);
        end

        if N_mtq > 0
            B_mtq = [zeros(3,N_mtq);
                     -(J \ (crossmat(B_vector) * A_mtq))];
        else
            B_mtq = zeros(n_x, 0);
        end

        B_sys = [B_rw, B_mtq];
    end

    % If no actuators, rank is zero for any nonzero task dimension
    if isempty(B_sys) || size(B_sys,2) == 0
        P = build_task_projection(task_mode, control_h, N_rw, a_b);
        n = size(P,1);
        r = 0;
        return;
    end

    % Column scaling
    S_rw  = 1.0;
    S_mtq = 1e6;
    Scale = diag([repmat(S_rw, 1, N_rw), repmat(S_mtq, 1, N_mtq)]);
    B_scaled = B_sys * Scale;

    % Controllability matrix
    Co = ctrb(A_sys, B_scaled);

    % Task projection z = P x, evaluate rank(P*Co)
    P = build_task_projection(task_mode, control_h, N_rw, a_b);
    n = size(P,1);

    Co_z = P * Co;
    if isempty(Co_z)
        r = 0;
        return;
    end

    svals = svd(Co_z);
    if isempty(svals) || max(svals) == 0
        r = 0;
        return;
    end
    tol = max(size(Co_z)) * eps(max(svals));
    r = sum(svals > tol);
end

% -------------------------------------------------------------------------
function P = build_task_projection(task_mode, control_h, N_rw, a_b)
% Returns P such that z = P x is the required controllable subspace.

    a = a_b(:);
    if norm(a) < eps
        a = [0;0;1];
    end
    a = a / norm(a);

    % Orthonormal basis U_perp spanning a^\perp (3x2)
    if abs(a(1)) < 0.9
        tmp = [1;0;0];
    else
        tmp = [0;1;0];
    end
    u1 = cross(a, tmp); u1 = u1 / norm(u1);
    u2 = cross(a, u1);  u2 = u2 / norm(u2);
    UperpT = [u1.'; u2.']; % 2x3

    task_mode = lower(string(task_mode));

    if control_h
        % x = [sigma(3); omega(3); h(N_rw)]
        switch task_mode
            case "full"
                P = eye(6 + N_rw);

            case "vector"
                % alignment (2) + transverse rate damping (2) + h
                P = [UperpT, zeros(2,3), zeros(2,N_rw);
                     zeros(2,3), UperpT, zeros(2,N_rw);
                     zeros(N_rw,3), zeros(N_rw,3), eye(N_rw)];

            case {"vector_damping","vector+damping","vector_damp"}
                % alignment (2) + full rate damping (3) + h
                P = [UperpT, zeros(2,3), zeros(2,N_rw);
                     zeros(3,3), eye(3), zeros(3,N_rw);
                     zeros(N_rw,3), zeros(N_rw,3), eye(N_rw)];

            otherwise
                error("Unknown task_mode. Use 'full', 'vector', or 'vector_damping'.");
        end

    else
        % x = [sigma(3); omega(3)]
        switch task_mode
            case "full"
                P = eye(6);

            case "vector"
                % alignment (2) + transverse rate damping (2)
                P = [UperpT, zeros(2,3);
                     zeros(2,3), UperpT];

            case {"vector_damping","vector+damping","vector_damp"}
                % alignment (2) + full rate damping (3)
                P = [UperpT, zeros(2,3);
                     zeros(3,3), eye(3)];

            otherwise
                error("Unknown task_mode. Use 'full', 'vector', or 'vector_damping'.");
        end
    end
end

function [r, n] = ltv_rank(J, A_rw, D_rw, h_0, A_mtq, B_fun, tf, control_h, task_mode, a_b)
%LTV_RANK  Rank test for LTV controllability of a task-relevant subspace z = P x via Gramian.
% Returns:
%   r : rank of projected Gramian Wz = P*Wc*P'
%   n : dimension of task-relevant subspace z \in R^n
%
% Inputs:
%   J         (3x3) inertia
%   A_rw      (3xN_rw) RW axes (may be [])
%   D_rw      (N_rw x N_rw) RW inertia matrix (may be [])
%   h_0       (N_rw x 1) nominal wheel momentum (may be [])
%   A_mtq     (3xN_mtq) MTQ axes (may be [])
%   B_fun     function handle @(t) -> (3x1) B-field vector
%   tf        final integration time
%   control_h (logical) include wheel momentum in state of interest and dynamics
%   task_mode (char/string) 'full' | 'vector' | 'vector_damping'
%     - 'vector'         : alignment + transverse rate damping
%     - 'vector_damping' : alignment + full rate damping (all 3 components)
%   a_b       (3x1, optional) body-fixed unit axis for vector pointing (default [0;0;1])

    if nargin < 10 || isempty(a_b),      a_b = [0;0;1];      end
    if nargin < 9  || isempty(task_mode), task_mode = "full"; end
    if nargin < 8  || isempty(control_h), control_h = false; end

    N_rw  = size(A_rw,  2);
    N_mtq = size(A_mtq, 2);

    % Safe empties
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

    % Task projection z = P x
    P = build_task_projection(task_mode, control_h, N_rw, a_b);
    n = size(P,1);

    % If no actuators at all, nothing is controllable
    if (N_rw + N_mtq) == 0
        r = 0;
        return;
    end

    crossmat = @(v) [   0   -v(3)  v(2);
                      v(3)    0   -v(1);
                     -v(2)  v(1)    0 ];

    % Column scaling (conditioning)
    S_rw  = 1.0;
    S_mtq = 1e6;
    Scale = diag([repmat(S_rw, 1, N_rw), repmat(S_mtq, 1, N_mtq)]);

    % ----------------------------
    % Build LTV system (A constant here, B time-varying through B_fun)
    % ----------------------------
    if control_h
        n_x = 6 + N_rw;

        A_core = [zeros(3,3), 0.25*eye(3);
                  zeros(3,3), J \ crossmat(H_rw_body)];
        A_top_right = zeros(6, N_rw);
        A_bot_left  = [zeros(N_rw,3), -D_rw*A_rw'*(J \ crossmat(H_rw_body))];
        A_bot_right = zeros(N_rw, N_rw);

        A_sys = [A_core, A_top_right;
                 A_bot_left, A_bot_right];

        B_phys = @(t) build_B_full(J, A_rw, D_rw, A_mtq, B_fun, t, crossmat);
    else
        n_x = 6;

        A_sys = [zeros(3,3), 0.25*eye(3);
                 zeros(3,3), J \ crossmat(H_rw_body)];

        B_phys = @(t) build_B_reduced(J, A_rw, A_mtq, B_fun, t, crossmat);
    end

    B_scaled = @(t) B_phys(t) * Scale;

    % If inputs exist but B is always zero, rank will be zero
    % (we let the integration handle it)

    % ----------------------------
    % Integrate Phi and Gramian Wc
    % ----------------------------
    z0 = [reshape(eye(n_x), [], 1); zeros(n_x^2, 1)];
    opts = odeset('RelTol', 1e-12, 'AbsTol', 1e-14);

    [~, z] = ode45(@(t,zz) ode_gramian(t, zz, n_x, A_sys, B_scaled), [0, tf], z0, opts);

    Wc = reshape(z(end, n_x^2+1:end), n_x, n_x);
    Wc = 0.5*(Wc + Wc.');

    % Project to task subspace
    Wz = P * Wc * P.';
    Wz = 0.5*(Wz + Wz.');

    % Rank via SVD (robust to scaling)
    svals = svd(Wz);
    if isempty(svals) || max(svals) == 0
        r = 0;
        return;
    end
    tol = max(size(Wz)) * eps(max(svals));
    r = sum(svals > tol);
end

% -------------------------------------------------------------------------
function dz = ode_gramian(t, z, n, A, B_fun)
    Phi = reshape(z(1:n^2), n, n);
    B = B_fun(t);
    if isempty(B)
        BBt = zeros(n);
    else
        BBt = B * B.';
    end
    Phi_dot = A * Phi;
    W_dot   = Phi * BBt * Phi.';
    dz = [Phi_dot(:); W_dot(:)];
end

% -------------------------------------------------------------------------
function B = build_B_full(J, A_rw, D_rw, A_mtq, B_fun, t, crossmat)
    N_rw  = size(A_rw,2);
    N_mtq = size(A_mtq,2);

    b = B_fun(t);
    if isempty(b), b = zeros(3,1); end

    % [sigma; omega; h] rows
    top    = [zeros(3,N_rw), zeros(3,N_mtq)];
    mid    = [J\A_rw,        -(J\(crossmat(b)*A_mtq))];
    bot_rw = -eye(N_rw) - D_rw*A_rw'*(J\A_rw);
    bot_mt =  D_rw*A_rw'*(J\(crossmat(b)*A_mtq));
    bot    = [bot_rw, bot_mt];

    B = [top; mid; bot];
end

% -------------------------------------------------------------------------
function B = build_B_reduced(J, A_rw, A_mtq, B_fun, t, crossmat)
    N_rw  = size(A_rw,2);
    N_mtq = size(A_mtq,2);

    b = B_fun(t);
    if isempty(b), b = zeros(3,1); end

    % [sigma; omega] rows
    top = [zeros(3,N_rw), zeros(3,N_mtq)];
    bot = [J\A_rw,        -(J\(crossmat(b)*A_mtq))];

    B = [top; bot];
end

% -------------------------------------------------------------------------
function P = build_task_projection(task_mode, control_h, N_rw, a_b)
    a = a_b(:);
    if norm(a) < eps, a = [0;0;1]; end
    a = a / norm(a);

    % Orthonormal basis for a^\perp
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

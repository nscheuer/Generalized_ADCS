%% Mega Table Generator: Controllability Analysis
clear; clc;

% --- Physical Constants ---
J_cage = diag([1; 1.2; 0.9]); 
tf = 600;

VARYING_B_FIELD = true;
B0_phys = 1e-6; 
if VARYING_B_FIELD
    period = 5400;
    B_fun = @(t) B0_phys * [cos(2*pi*t/period); sin(2*pi*t/period); 0.5*cos(2*pi*t/(period/2))];
else
    dir = [1; 0.5; 0.2];
    B_fun = @(t) B0_phys * (dir / norm(dir));
end

% --- Task Definitions ---
% Mini-table columns
modes      = ["full", "vector", "vector_damping"]; 
mode_names = ["F", "V", "VD"]; % Short names for print width
% Mini-table rows
use_h      = [false, true]; 
h_labels   = ["h-", "h+"];

% --- Axis Definitions ---
% We define the pool of axes based on your instructions
% MTQ Sequence: 0 -> (100) -> (100, 010) -> (100, 010, 001)
mtq_pool = [1, 0, 0; 
            0, 1, 0; 
            0, 0, 1]'; 

% RW Sequence: 0 -> (001) -> (001, 010) -> (001, 010, 100) -> (..., 111)
rw_pool = [0, 0, 1;
           0, 1, 0;
           1, 0, 0;
           1, 1, 1]';
% Normalize the 1,1,1 vector for physical consistency
rw_pool(:,4) = rw_pool(:,4) / norm(rw_pool(:,4));

%% --- Main Calculation Loop ---
fprintf('<strong>LTV MEGA TABLE</strong>\n');
fprintf('Legend for Mini-Cells:\n');
fprintf('   [ h- | Full  Vec  VecD ] (h not included)\n');
fprintf('   [ h+ | Full  Vec  VecD ] (h included)\n');
fprintf('Values are formatted as: rank / n\n\n');

% Header Row
fprintf('%-8s |', 'MTQ\RW');
for j = 0:4
    fprintf(' %-22s |', sprintf('RW=%d', j));
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 8 + 3 + (24*5)));

% Loop Rows (MTQ Count)
for n_mtq = 0:3
    
    % Prepare the two text lines for this MTQ row (h- and h+)
    line_h_false = sprintf(' %-6s |', sprintf('MTQ=%d', n_mtq)); 
    line_h_true  = sprintf(' %-6s |', ''); 
    
    % Loop Columns (RW Count)
    for n_rw = 0:4
        
        % 1. Build Matrices for this specific cell
        current_A_mtq = mtq_pool(:, 1:n_mtq);
        current_A_rw  = rw_pool(:, 1:n_rw);
        
        % Handle variable RW inertia/momentum sizes
        if n_rw > 0
            current_D_rw = 0.1 * eye(n_rw);
            current_h_0  = zeros(n_rw, 1);
        else
            current_D_rw = [];
            current_h_0  = [];
        end
        
        % 2. Calculate the Mini-Table (2x3)
        % Stores strings like "6/6"
        res_strs = strings(2, 3); 
        
        for h_idx = 1:2 % Row of mini-table (h false/true)
            for m_idx = 1:3 % Col of mini-table (Full, Vec, VecD)
                
                [r, n] = ltv_rank(J_cage, current_A_rw, current_D_rw, current_h_0, ...
                                  current_A_mtq, B_fun, tf, ...
                                  use_h(h_idx), modes(m_idx));
                
                res_strs(h_idx, m_idx) = sprintf('%d/%d', r, n);
            end
        end
        
        % 3. Format strings for display
        % Format: "h-: F V VD"
        str_false = sprintf('h-: %-4s %-4s %-4s', res_strs(1,1), res_strs(1,2), res_strs(1,3));
        str_true  = sprintf('h+: %-4s %-4s %-4s', res_strs(2,1), res_strs(2,2), res_strs(2,3));
        
        line_h_false = [line_h_false, sprintf(' %-22s |', str_false)]; %#ok<AGROW>
        line_h_true  = [line_h_true,  sprintf(' %-22s |', str_true)];  %#ok<AGROW>
    end
    
    % Print the Mega-Row
    fprintf('%s\n', line_h_false);
    fprintf('%s\n', line_h_true);
    fprintf('%s\n', repmat('-', 1, 8 + 3 + (24*5)));
end
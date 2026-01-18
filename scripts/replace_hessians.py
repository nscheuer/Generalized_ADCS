#!/usr/bin/env python3
import re
from pathlib import Path
p = Path('trajectory_planner/src/planner/main.cpp')
text = p.read_text()
backup = p.with_suffix('.cpp.bak')
backup.write_text(text)

# Helper to replace a block between start_idx and end_idx with replacement
def replace_range(s, start, end, repl):
    return s[:start] + repl + s[end:]

# Find all occurrences of 'arma::mat ddf__dxdx' and process
patterns = [
    ('ddx','arma::mat ddf__dxdx'),
    ('ddu','arma::mat ddf__dudu'),
    ('dudx','arma::mat ddf__dudx'),
]

changed = 0
for kind, key in patterns:
    idx = 0
    while True:
        i = text.find(key, idx)
        if i == -1:
            break
        # find line end
        line_end = text.find('\n', i)
        # find block end by searching for next CHECK/REQUIRE that mentions the corresponding expected matrix
        # For ddf__dxdx look for 'lkxx' or 'ddf__dxdxQ' or 'REQUIRE'/'CHECK' nearby
        search_after = text[line_end:line_end+2000]
        # try to find 'CHECK' or 'REQUIRE' after the block which references lkxx/lkuu/lkux
        m = re.search(r'\n\s*(?:cout<<|CHECK\(|REQUIRE\(|arma::mat ddf__dxdxQ|arma::mat ddf__dudxQ|REQUIRE\(|CHECK\()', text[line_end:])
        if m:
            end_search = line_end + m.start()
        else:
            # fallback: find next blank line after some braces
            end_search = text.find('\n\n', line_end)
            if end_search == -1:
                end_search = len(text)
        block = text[i:end_search]
        # attempt to find inner return expression from first 'auto f' lambda
        mfx = re.search(r'auto\s+fx[\w]*\s*=\s*\[=,&costset_tmp\]\s*\(double\s+xi\)\s*\{\s*return\s+([^;]+);', block)
        if not mfx:
            # try other patterns
            mfx = re.search(r'auto\s+fxi\s*=\s*\[=,&costset_tmp\]\s*\(double\s+xi\)\s*\{\s*return\s+([^;]+);', block)
        if not mfx:
            mfx = re.search(r'auto\s+fxi\s*=\s*\[=,&costset_tmp\]\s*\(double\s+xi\)\s*\{\s*([^}]+)\}', block, re.S)
        if not mfx:
            # cannot parse, skip
            idx = i + len(key)
            continue
        expr = mfx.group(1).strip()
        # prepare replacement based on kind
        if 'stepcost_vec' in expr:
            repl = '\n\t\t// compute Hessian wrt state using numerical_hessian helper\n\t\tauto f_x = [=,&costset_tmp,&sat,&uk,&z3,&satvec_k,&ECIvec_k,&BECI_k](const arma::vec& xv){\n\t\t\treturn sat.stepcost_vec(k, N, xv, uk, z3, satvec_k, ECIvec_k, BECI_k, &costset_tmp);\n\t\t};\n\t\tarma::mat ddf__dxdx = numerical_hessian(f_x, xk, 0.0, 7);\n'
        elif 'stepcost_quat' in expr:
            repl = '\n\t\t// compute Hessian wrt state using numerical_hessian helper\n\t\tauto f_x = [=,&costset_tmp,&sat,&uk,&z3,&satvec_k,&ECIvec_k,&BECI_k](const arma::vec& xv){\n\t\t\treturn sat.stepcost_quat(k, N, xv, uk, z3, satvec_k, ECIvec_k, BECI_k, &costset_tmp);\n\t\t};\n\t\tarma::mat ddf__dxdx = numerical_hessian(f_x, xk, 0.0, 7);\n'
        elif 'getConstraints' in expr or 'dot(eind' in expr:
            # assume expression like arma::dot(eind,sat.getConstraints(k,N,uk,xk + ...))
            repl = '\n\t\t// compute Hessian wrt state using numerical_hessian helper for constraint output\n\t\tauto f_x = [=,&costset_tmp,&sat,&uk,&sunk](const arma::vec& xv){\n\t\t\treturn arma::dot(eind, sat.getConstraints(k, N, uk, xv, sunk));\n\t\t};\n\t\tarma::mat ddf__dxdx = numerical_hessian(f_x, xk, 0.0, 7);\n'
        elif 'dynamics_pure' in expr or 'rk4' in expr or 'state_norm' in expr:
            # map to generic dynamics call replacing xk with xv
            # detect function name
            func_m = re.search(r'([\w:]+)\s*\(', expr)
            func = func_m.group(1) if func_m else None
            if func and 'state_norm' in func:
                repl = '\n\t\tauto f_x = [=,&costset_tmp,&sat](const arma::vec& xv){\n\t\t\treturn arma::dot(eind, sat.state_norm(xv));\n\t\t};\n\t\tarma::mat ddf__dxdx = numerical_hessian(f_x, xk, 0.0, 7);\n'
            else:
                repl = '\n\t\t// compute Hessian wrt state using numerical_hessian helper\n\t\tauto f_x = [=,&costset_tmp,&sat,&uk,&dynamics_info_k](const arma::vec& xv){\n\t\t\treturn arma::dot(eind, ' + func + '(xv, uk, dynamics_info_k));\n\t\t};\n\t\tarma::mat ddf__dxdx = numerical_hessian(f_x, xk, 0.0, 7);\n'
        else:
            # fallback: try a generic replacement that wraps the original expr substituting xk->xv
            expr2 = expr.replace('xk', 'xv')
            repl = '\n\t\tauto f_x = [=,&costset_tmp,&sat](const arma::vec& xv){\n\t\t\treturn ' + expr2 + ';\n\t\t};\n\t\tarma::mat ddf__dxdx = numerical_hessian(f_x, xk, 0.0, 7);\n'
        # apply replacement
        text = replace_range(text, i, end_search, repl)
        changed += 1
        idx = i + len(repl)

# write back if changed
if changed > 0:
    p.write_text(text)
    print(f"Applied {changed} replacements and wrote {p}")
else:
    print("No replacements applied")

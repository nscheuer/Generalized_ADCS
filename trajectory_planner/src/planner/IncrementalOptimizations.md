# Incremental ALTRO Optimizations

Apply these one at a time, benchmark after each.

---

## Optimization 1: Pre-extract columns (SAFEST, ~5% speedup)

**File:** `OldPlanner.cpp`, function `backwardPass()` around line 1900

**Current code pattern:**
```cpp
while(k >= 0) {
    xk = Xset.col(k);
    // ... later ...
    dynamics_info_k = make_tuple(Bset.col(k), Rset.col(k), ...);
    // ... later ...
    costJac = sat.veccostJacobians(..., Bset.col(k), ...);
```

**Change to:**
```cpp
// BEFORE the while loop, add:
// Pre-extract frequently accessed columns
std::vector<arma::vec> x_cache(N), u_cache(N);
std::vector<arma::vec3> B_cache(N), R_cache(N), V_cache(N), S_cache(N), sat_cache(N);
std::vector<arma::vec> eci_cache(N), lam_cache(N), muk_cache(N);

for(int i = 0; i < N; i++) {
    x_cache[i] = Xset.col(i);
    u_cache[i] = Uset.col(i);
    B_cache[i] = Bset.col(i);
    R_cache[i] = Rset.col(i);
    V_cache[i] = Vset.col(i);
    S_cache[i] = sunset.col(i);
    sat_cache[i] = satvec.col(i);
    eci_cache[i] = ECIvec.col(i);
    lam_cache[i] = lambdaSet.col(i);
    muk_cache[i] = muSet.col(i);
}

// In the while loop, replace:
//   xk = Xset.col(k);
// with:
//   xk = x_cache[k];
// etc.
```

---

## Optimization 2: Pre-allocate workspace matrices (SAFE, ~10% speedup)

**File:** `OldPlanner.cpp`, function `backwardPass()` around line 1900

**Current code pattern:**
```cpp
while(k >= 0) {
    // These allocate memory every iteration!
    Aqk = Aqk.zeros();
    Bqk = Bqk.zeros();
```

**Change to:**
```cpp
// BEFORE the while loop, the matrices are already declared.
// Just make sure they're sized correctly once:
const int nx = sat.reduced_state_N();
const int nu = sat.control_N();

Aqk.set_size(nx, nx);
Bqk.set_size(nx, nu);
// ... etc for other matrices

// In the while loop, use .zeros() which just fills with zeros, doesn't reallocate
// (this is already mostly correct, but ensure sizes are set before loop)
```

**Better approach - use .fill(0) pattern:**
```cpp
// Replace:
Aqk = Aqk.zeros();  // May reallocate

// With:
Aqk.zeros();  // In-place zero (but .zeros() on sized matrix should be fine)

// Or more explicitly:
Aqk.fill(0.0);  // Guaranteed no reallocation
```

---

## Optimization 3: Cache rotation matrices (MODERATE, ~15% speedup)

**File:** `OldPlanner.cpp`, function `backwardPass()`

**Current:** `rotMat(qk)` is called multiple times for the same quaternion (once directly, 
then again inside `dRTBdq`, `dRTBdqQ`, etc.)

**Change:** Compute once at start of each k iteration:

```cpp
// At start of while loop body, after getting xk:
xk = x_cache[k];
qk = xk.rows(3, 6);
qk = arma::normalise(qk);  // Ensure normalized

// Cache these for this timestep
const arma::mat33 Rk = rotMat(qk);
const arma::mat33 RkT = Rk.t();

// Then pass RkT to functions that need it, or modify those functions
// to accept a cached rotation matrix
```

**Note:** This requires modifying `Satellite::veccostJacobians()` etc. to optionally
accept pre-computed rotation matrices. Can be done with overloads or optional params.

---

## Optimization 4: Fuse intermediate matrix products (MODERATE, ~10% speedup)

**File:** `OldPlanner.cpp`, function `backwardPass()` around line 2060

**Current code:**
```cpp
Qkxx = costJac.lxx + trans(Aqk)*Pk*Aqk + trans(ckx)*Imuk*ckx;
Qkux = costJac.lux + trans(Bqk)*Pk*Aqk + trans(cku)*Imuk*ckx;
Qkuu = costJac.luu + trans(Bqk)*Pk*Bqk + trans(cku)*Imuk*cku;
```

**Change to:**
```cpp
// Compute shared products once
const arma::mat PkAqk = Pk * Aqk;           // nx x nx, used in Qkxx and Qkux
const arma::mat BqkT = Bqk.t();             // nu x nx
const arma::mat BqkT_Pk = BqkT * Pk;        // nu x nx, used in Qkux and Qkuu
const arma::mat ImukCkx = Imuk * ckx;       // nc x nx, used in Qkxx and Qkux
const arma::mat ckuT_Imuk = cku.t() * Imuk; // nx x nc, used in Qkux and Qkuu

// Now build Q matrices
Qkxx = costJac.lxx + Aqk.t() * PkAqk + ckx.t() * ImukCkx;
Qkux = costJac.lux + BqkT_Pk * Aqk + ckuT_Imuk * ckx;
Qkuu = costJac.luu + BqkT_Pk * Bqk + ckuT_Imuk * cku;
```

---

## Optimization 5: Reuse Cholesky for both solves (SAFE, ~5% speedup)

**File:** `OldPlanner.cpp`, around line 2150

**Current code:**
```cpp
reset |= !chol(Qkuureg_chol, Qkuureg);
// ...
reset |= !solve(Kk, Qkuureg, Qkux, solve_opts::no_approx);
reset |= !solve(dk, Qkuureg, Qku, solve_opts::no_approx);
```

**Change to:**
```cpp
reset |= !chol(Qkuureg_chol, Qkuureg);
if(!reset) {
    // Solve using Cholesky: Qkuureg * x = b  where Qkuureg = L * L^T
    // This is faster than general solve since we already have the factorization
    
    // For Kk: solve L * y = Qkux, then L^T * Kk = y
    arma::mat y_K = arma::solve(arma::trimatl(Qkuureg_chol), Qkux, 
                                 arma::solve_opts::fast);
    Kk = arma::solve(arma::trimatu(Qkuureg_chol.t()), y_K, 
                     arma::solve_opts::fast);
    
    // For dk: solve L * y = Qku, then L^T * dk = y  
    arma::vec y_d = arma::solve(arma::trimatl(Qkuureg_chol), Qku,
                                 arma::solve_opts::fast);
    dk = arma::solve(arma::trimatu(Qkuureg_chol.t()), y_d,
                     arma::solve_opts::fast);
    
    reset |= (Kk.has_nan() || Kk.has_inf() || dk.has_nan() || dk.has_inf());
}
```

---

## Optimization 6: Add timing instrumentation (FOR PROFILING)

**File:** `OldPlanner.cpp`, in `backwardPass()` and `forwardPass()`

```cpp
// At top of file, add:
#include <chrono>

// In backwardPass, wrap main loop:
auto bp_start = std::chrono::high_resolution_clock::now();

while(k >= 0) {
    // ... existing code ...
}

auto bp_end = std::chrono::high_resolution_clock::now();
if(verbose_level >= 2) {
    double ms = std::chrono::duration<double, std::milli>(bp_end - bp_start).count();
    std::cout << "[TIMING] Backward pass: " << ms << " ms (N=" << N << ")" << std::endl;
}
```

---

## Testing procedure

1. Run existing tests to establish baseline timing
2. Apply Optimization 1, run tests, record timing
3. Apply Optimization 2, run tests, record timing
4. ... etc

Keep a log:
```
Baseline:           XX.X ms average backward pass
After Opt 1:        XX.X ms (Y% improvement)
After Opt 1+2:      XX.X ms (Z% cumulative)
...
```

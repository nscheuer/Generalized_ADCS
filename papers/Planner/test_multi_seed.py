#!/usr/bin/env python
"""Quick multi-seed test for planner convergence."""
import subprocess
import sys

def main():
    seeds = [0, 1, 2, 3, 4, 5]
    results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print('='*60)
        
        # Run as subprocess to get clean state each time
        cmd = [sys.executable, "papers/Planner/generate_mc_3mtq+1rw_plan_full.py", "-t", "-s", str(seed)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Check for success markers in output
        output = proc.stdout + proc.stderr
        
        # Look for final angle error
        success = "Test run completed successfully" in output
        
        # Extract angle info from last diagnostic line
        lines = output.split('\n')
        last_diag = None
        for line in reversed(lines):
            if "Angle[" in line and "end:" in line:
                last_diag = line
                break
        
        if last_diag:
            # Parse end angle
            try:
                end_part = last_diag.split("end:")[1].split("°")[0]
                end_angle = float(end_part)
            except:
                end_angle = None
        else:
            end_angle = None
        
        status = "SUCCESS" if success else "FAILED"
        results.append((seed, status, end_angle, proc.returncode))
        
        print(f"  Status: {status}")
        print(f"  End angle: {end_angle}°" if end_angle is not None else "  End angle: N/A")
        if not success:
            print(f"  Return code: {proc.returncode}")
            # Print last few lines on failure
            print("  Last output lines:")
            for line in lines[-10:]:
                if line.strip():
                    print(f"    {line}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    success_count = sum(1 for _, s, _, _ in results if s == "SUCCESS")
    for seed, status, end_angle, _ in results:
        angle_str = f"{end_angle:.1f}°" if end_angle is not None else "N/A"
        print(f"  Seed {seed}: {status} (end angle: {angle_str})")
    
    print(f"\nTotal: {success_count}/{len(results)} succeeded")
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

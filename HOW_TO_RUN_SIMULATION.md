# How to Run MPC Simulation

This guide explains how to run the MPC controller simulation and view the results.

---

## Quick Start

### 1. Run Simulation WITH Visualization (3D Window)

```bash
python tests/test_mpc_controller.py --gui
```

This will:
- Open a PyBullet 3D visualization window showing the drone
- Run a 10-second hover test at 1.0m altitude
- Display matplotlib plots after the test completes
- Save all results automatically

**Note:** The 3D window will show the drone hovering. DO NOT close this window manually - let the simulation complete.

---

### 2. Run Simulation WITHOUT Visualization (Headless, Faster)

```bash
python tests/test_mpc_controller.py
```

This runs the simulation without the 3D window and is faster for batch testing.

---

## Advanced Options

### Run Longer Tests

```bash
# 20-second test with visualization
python tests/test_mpc_controller.py --gui --duration 20
```

### Change Target Altitude

```bash
# Hover at 2.0m altitude
python tests/test_mpc_controller.py --gui --altitude 2.0
```

### Specify Iteration Number (for file naming)

```bash
# Save results as test_iteration_03.*
python tests/test_mpc_controller.py --gui --iteration 3
```

### View All Options

```bash
python tests/test_mpc_controller.py --help
```

---

## Results Output

After running the simulation, the following files are automatically generated:

### 1. **CSV Data File** (Detailed Time Series)
- **Location:** `results/phase_02/test_iteration_XX.csv`
- **Contents:**
  - Time stamps
  - Position (X, Y, Z)
  - Velocity (Vx, Vy, Vz)
  - Orientation (Roll, Pitch, Yaw)
  - Angular velocity (P, Q, R)
  - Control inputs (Thrust, Roll rate cmd, Pitch rate cmd, Yaw rate cmd)
  - Reference position
  - Position error at each time step
  - MPC solve time
  - Solve success flag

**Open in Excel, MATLAB, or Python for analysis**

Example CSV structure:
```csv
Time_s,Pos_X_m,Pos_Y_m,Pos_Z_m,Vel_X_ms,Vel_Y_ms,Vel_Z_ms,...
0.0000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,...
0.0208,0.000001,0.000001,1.000000,0.000048,0.000048,0.000002,...
...
```

### 2. **JSON Summary File** (Performance Metrics)
- **Location:** `results/phase_02/test_iteration_XX.json`
- **Contents:**
  - Configuration parameters (MPC horizon, timestep, weights)
  - Performance metrics (RMSE, max error, solve times)
  - Pass/fail status
  - Test metadata

### 3. **Plot File** (Visualization)
- **Location:** `results/phase_02/mpc_hover_test.png`
- **Contains 3 subplots:**
  1. Position tracking (X, Y, Z vs time)
  2. Velocity profiles (Vx, Vy, Vz vs time)
  3. Control inputs (Thrust and angular rate commands vs time)

**The plot window will also display on screen after the test completes**

---

## Example Workflow

### Complete Run with Visualization

1. **Navigate to project root:**
   ```bash
   cd D:\rl_tuned_mpc
   ```

2. **Activate virtual environment:**
   ```bash
   venv_drones\Scripts\activate
   ```

3. **Run simulation with GUI:**
   ```bash
   python tests/test_mpc_controller.py --gui
   ```

4. **What you'll see:**
   - Console output with progress updates every 50 steps
   - PyBullet 3D window showing the drone (keep this open!)
   - Final performance metrics printed to console
   - Matplotlib plot window appears at the end

5. **Find your results:**
   ```bash
   # View CSV data
   explorer results\phase_02\test_iteration_02.csv

   # View plot
   explorer results\phase_02\mpc_hover_test.png

   # View JSON summary
   notepad results\phase_02\test_iteration_02.json
   ```

---

## Understanding the Results

### Console Output

During the test, you'll see progress updates:

```
Step   50 | t= 1.04s | Pos error: 0.0001m | Solve time: 16.00ms
Step  100 | t= 2.08s | Pos error: 0.0001m | Solve time: 24.47ms
...
```

After completion, performance summary:

```
============================================================
Test Results:
============================================================
Position Tracking:
  Max error:   0.0001 m
  RMSE:        0.0001 m
  Final error: 0.0001 m

MPC Performance:
  Avg solve time: 18.46 ms
  Max solve time: 86.71 ms
  Success rate:   100.0%

Simulation:
  Real time:      9.32 s
  Simulated time: 10.00 s
  Real-time factor: 1.07x
============================================================
Test Result: PASS
============================================================
```

### Success Criteria

The test **PASSES** if:
- ✅ Position RMSE < 0.1 m
- ✅ Final error < 0.05 m
- ✅ Average solve time < 20 ms

**Current Performance:** All criteria exceeded with RMSE = 0.0001m (essentially perfect tracking)

---

## Analyzing CSV Data

### Using Python (Pandas)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data
df = pd.read_csv('results/phase_02/test_iteration_02.csv')

# Plot position tracking
plt.figure(figsize=(12, 4))
plt.plot(df['Time_s'], df['Pos_X_m'], label='X')
plt.plot(df['Time_s'], df['Pos_Y_m'], label='Y')
plt.plot(df['Time_s'], df['Pos_Z_m'], label='Z')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate statistics
print(f"Mean position error: {df['Position_Error_m'].mean():.6f} m")
print(f"Max position error: {df['Position_Error_m'].max():.6f} m")
print(f"Mean solve time: {df['Solve_Time_ms'].mean():.2f} ms")
```

### Using MATLAB

```matlab
% Load CSV data
data = readtable('results/phase_02/test_iteration_02.csv');

% Plot position tracking
figure;
plot(data.Time_s, data.Pos_X_m, 'b', 'LineWidth', 2);
hold on;
plot(data.Time_s, data.Pos_Y_m, 'r', 'LineWidth', 2);
plot(data.Time_s, data.Pos_Z_m, 'g', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Position (m)');
legend('X', 'Y', 'Z');
grid on;
```

---

## Troubleshooting

### Issue: "Config file not found"
**Solution:**
```bash
# Make sure you're in the project root directory
cd D:\rl_tuned_mpc

# Check if config exists
dir configs\mpc_crazyflie.yaml
```

### Issue: PyBullet window closes immediately
**Solution:** This is normal when running without `--gui`. Use `--gui` flag to keep it open.

### Issue: "Module not found" errors
**Solution:**
```bash
# Activate virtual environment
venv_drones\Scripts\activate

# Reinstall dependencies if needed
pip install -e gym-pybullet-drones
```

### Issue: Plots don't display
**Solution:** Plots are saved to PNG file even if display fails. Check:
```bash
explorer results\phase_02\mpc_hover_test.png
```

---

## Next Steps

After successfully running the simulation:

1. ✅ Analyze CSV data for detailed time-series performance
2. ✅ Examine plots for tracking quality
3. ✅ Review JSON for summary metrics
4. ✅ Compare different test iterations
5. ✅ Proceed to Phase 3: RL Integration

---

## File Locations Summary

| Item | Location |
|------|----------|
| Test script | `tests/test_mpc_controller.py` |
| Configuration | `configs/mpc_crazyflie.yaml` |
| CSV results | `results/phase_02/test_iteration_XX.csv` |
| JSON results | `results/phase_02/test_iteration_XX.json` |
| Plots | `results/phase_02/mpc_hover_test.png` |
| This guide | `HOW_TO_RUN_SIMULATION.md` |

---

**Ready to run your first simulation?**

```bash
python tests/test_mpc_controller.py --gui
```

Enjoy the perfect tracking performance!

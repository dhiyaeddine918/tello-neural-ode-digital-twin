# Data-Driven Digital Twin of a DJI Tello Drone with Controlled Neural ODEs

Learning the continuous-time dynamics of a DJI Tello drone from real flight telemetry (collected through ROS 2) with a controlled Neural ODE, `dx/dt = f_θ(x, u)`, implemented in PyTorch.

*M1 final-year project, Université Paris-Saclay, 2026.*

## Problem

Can the dynamics of a small quadrotor be learned directly from flight data, without deriving aerodynamic and rigid-body equations, and be used as a simulator (digital twin) over many steps?

- **State** `x = [yaw, h, vgx, vgy, vgz]` (yaw, height, ground-frame velocities)
- **Control** `u = [cmd_lr, cmd_fb, cmd_ud, cmd_yaw]`
- **Task** predict the next state from the current state, the control and the time step `dt`

## Approach

The notebook follows the evolution of the project in four stages:

1. **Simulation prototype (Gazebo).** Point-to-point and square trajectories for training, circle for testing, with a first-order commanded-motion baseline (test RMSE x/y/z = 0.51 / 0.45 / 0.21).
2. **First real-data experiment.** 6 short Tello flights, Euler-style Neural ODE predictor (exploratory).
3. **Controlled Neural ODE.** MLP vector field with RK4 integration (variable `dt`), LayerNorm, AdamW, gradient clipping.
4. **Rollout training.** Trained on 10-step windows instead of one-step pairs, then evaluated in a 250-step open-loop rollout.

Final dataset: 4 flight recordings (forward/backward oscillation, forward, forward + yaw, yaw), 3617 raw samples, 3205 after cleaning (invalid ToF, height and velocity outliers, abnormal `dt`). Chronological 80/20 train/test split.

## Results

One-step prediction on the test set (normalized units, RMSE):

| State | Persistence baseline | One-step trained | Rollout trained |
|-------|---------------------:|-----------------:|----------------:|
| yaw   | **0.016** | 0.048 | 0.037 |
| h     | 0.045 | 0.052 | **0.041** |
| vgx   | 0.402 | 0.421 | **0.393** |
| vgy   | **0.140** | 0.146 | 0.160 |
| vgz   | **0.443** | 0.457 | 0.468 |

250-step open-loop rollout (RMSE, normalized units):

| State | One-step trained | Rollout trained | Change |
|-------|-----------------:|----------------:|-------:|
| yaw   | 0.937 | 0.912 | -2.6 % |
| h     | 0.799 | 0.815 | +2.0 % |
| vgx   | 0.245 | 0.234 | -4.5 % |
| vgy   | 0.177 | 0.140 | **-20.5 %** |
| vgz   | 1.068 | 1.075 | +0.7 % |

![Rollout, vgy](results/rollout_trained_vgy.png)

**Takeaways.** Rollout training helps on some states (clearest on `vgy`) but not on all. The persistence baseline (`x_{t+1} = x_t`) is very strong at a 0.1 s sampling interval, and the learned models do not beat it on every variable. Vertical velocity and height remain hard to predict.

## Limitations (honest notes)

- Small dataset (a few thousand samples, 4 flights); results are from a single run and a single split, with no confidence intervals.
- The 250-step rollout is compared between the two learned models only, not against a baseline rollout.
- The model overfits after about 125 epochs (test loss rises while train loss keeps decreasing); no early stopping is used.
- The scalers are fitted on the full dataset before the train/test split.
- The first-order model used as a baseline was evaluated on the simulation data (Part 1), not on the real Tello data.

## Future work

Early stopping and validation split, a rollout baseline, more diverse flights, physics-informed structure (rigid-body prior + learned residual), and noise-robust training.

## Repository structure

```
notebooks/   main notebook (Google Colab)
data/        Tello telemetry CSV files (see data/README.md)
results/     figures exported from the notebook
```

## How to run

```bash
git clone https://github.com/YOUR_USERNAME/tello-neural-ode-digital-twin.git
cd tello-neural-ode-digital-twin
pip install -r requirements.txt
jupyter notebook notebooks/tello_neural_ode_digital_twin.ipynb
```

The notebook was developed on Google Colab and reads the data from Google Drive. To run it locally, put the CSV files in `data/` and change `DATA_PATH` in Part 4 to `"../data"`.

## Tools

Python, PyTorch, torchdiffeq, NumPy, pandas, scikit-learn, Matplotlib, ROS 2 (data collection), Gazebo (simulation prototype).

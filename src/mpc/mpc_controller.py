"""
Nonlinear MPC Controller for Quadrotor
Uses CasADi for symbolic computation and IPOPT for optimization

State: [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
Control: [thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]

Author: Dr. Abdul Manan Khan
Project: RL-Enhanced MPC for Multi-Drone Systems
"""

import casadi as ca
import numpy as np
import yaml
from typing import Tuple, Optional
import time


class MPCController:
    """
    Nonlinear MPC for quadrotor trajectory tracking

    The controller solves a finite-horizon optimal control problem at each timestep
    to compute optimal control inputs that minimize tracking error while respecting
    system dynamics and control constraints.

    State vector (12D): [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
    Control vector (4D): [thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]
    """

    def __init__(self, config_path: str):
        """
        Initialize MPC controller with configuration

        Args:
            config_path: Path to YAML configuration file
        """
        self.load_config(config_path)
        self.setup_optimization()

        # Statistics
        self.solve_count = 0
        self.total_solve_time = 0
        self.failures = 0

    def load_config(self, config_path: str):
        """
        Load MPC parameters from YAML configuration file

        Args:
            config_path: Path to configuration YAML
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Prediction horizon
        self.N = config['mpc']['prediction_horizon']
        self.dt = config['mpc']['timestep']

        # Weight matrices (these will be tuned by RL)
        Q_diag = config['mpc']['Q']  # 12-dimensional
        R_diag = config['mpc']['R']  # 4-dimensional

        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)

        # Terminal cost weight (typically larger than Q)
        if 'Q_terminal' in config['mpc']:
            Q_term_diag = config['mpc']['Q_terminal']
            self.Q_terminal = np.diag(Q_term_diag)
        else:
            self.Q_terminal = 10.0 * self.Q  # Default: 10x stage cost

        # Control constraints
        self.u_min = np.array(config['mpc']['u_min'])
        self.u_max = np.array(config['mpc']['u_max'])

        # Drone parameters (platform-specific)
        self.mass = config['drone']['mass']
        self.Ixx = config['drone']['inertia']['Ixx']
        self.Iyy = config['drone']['inertia']['Iyy']
        self.Izz = config['drone']['inertia']['Izz']

        # Gravity constant
        self.g = 9.81

    def setup_optimization(self):
        """
        Setup CasADi optimization problem
        Creates symbolic variables, dynamics, cost function, and constraints
        """
        # State and control dimensions
        nx = 12  # [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
        nu = 4   # [thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]

        # Create symbolic variables for dynamics function
        x = ca.SX.sym('x', nx)
        u = ca.SX.sym('u', nu)

        # Drone dynamics (symbolic)
        x_dot = self._dynamics(x, u)

        # Create CasADi function for dynamics
        self.f_dynamics = ca.Function('f', [x, u], [x_dot],
                                       ['x', 'u'], ['x_dot'])

        # Setup the NLP (Nonlinear Programming) problem
        self._setup_nlp()

    def _dynamics(self, x: ca.SX, u: ca.SX) -> ca.SX:
        """
        Quadrotor dynamics (continuous time)

        Implements the nonlinear equations of motion for a quadrotor UAV.
        Uses simplified dynamics with:
        - 6-DOF rigid body dynamics
        - First-order actuator model
        - Euler angle representation

        Args:
            x: State vector [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
            u: Control vector [thrust, p_cmd, q_cmd, r_cmd]

        Returns:
            x_dot: State derivatives
        """
        # Extract states
        pos = x[0:3]      # Position [px, py, pz]
        vel = x[3:6]      # Velocity [vx, vy, vz]
        att = x[6:9]      # Euler angles [roll, pitch, yaw]
        omega = x[9:12]   # Angular velocity [p, q, r]

        # Extract controls
        thrust = u[0]         # Total thrust
        omega_cmd = u[1:4]    # Commanded angular velocities

        # Position derivatives = velocity
        pos_dot = vel

        # Velocity derivatives (Newton's 2nd law in world frame)
        phi, theta, psi = att[0], att[1], att[2]

        # Rotation matrix elements (body to world)
        # Using full rotation matrix for accuracy
        c_phi = ca.cos(phi)
        s_phi = ca.sin(phi)
        c_theta = ca.cos(theta)
        s_theta = ca.sin(theta)
        c_psi = ca.cos(psi)
        s_psi = ca.sin(psi)

        # Thrust force in world frame
        # Assuming thrust is along body z-axis
        thrust_world = ca.vertcat(
            thrust / self.mass * (s_psi * s_phi + c_psi * s_theta * c_phi),
            thrust / self.mass * (-c_psi * s_phi + s_psi * s_theta * c_phi),
            thrust / self.mass * (c_theta * c_phi) - self.g
        )

        vel_dot = thrust_world

        # Attitude kinematics (Euler angle rates)
        # Relationship between Euler angle rates and body angular velocities
        p, q, r = omega[0], omega[1], omega[2]

        att_dot = ca.vertcat(
            p + s_phi * ca.tan(theta) * q + c_phi * ca.tan(theta) * r,
            c_phi * q - s_phi * r,
            s_phi / ca.cos(theta) * q + c_phi / ca.cos(theta) * r
        )

        # Angular acceleration (simplified first-order actuator model)
        # Assumes fast inner-loop attitude control
        # tau = 10.0 -> 0.1s time constant
        tau = 10.0
        omega_dot = tau * (omega_cmd - omega)

        return ca.vertcat(pos_dot, vel_dot, att_dot, omega_dot)

    def _setup_nlp(self):
        """
        Setup the Nonlinear Programming problem for MPC

        Creates decision variables, parameters, objective function,
        and constraints for the MPC optimization problem.
        """
        nx = 12
        nu = 4

        # Create Opti object (CasADi optimization interface)
        self.opti = ca.Opti()

        # Decision variables
        # State trajectory: X = [x_0, x_1, ..., x_N]
        # Control trajectory: U = [u_0, u_1, ..., u_{N-1}]
        self.X = self.opti.variable(nx, self.N + 1)
        self.U = self.opti.variable(nu, self.N)

        # Parameters (will be set at each MPC iteration)
        self.X0 = self.opti.parameter(nx)              # Initial state
        self.X_ref = self.opti.parameter(nx, self.N + 1)  # Reference trajectory

        # Objective function
        obj = 0

        # Stage cost (running cost)
        for k in range(self.N):
            # State tracking error
            x_err = self.X[:, k] - self.X_ref[:, k]
            obj += ca.mtimes([x_err.T, self.Q, x_err])

            # Control effort penalty
            obj += ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])

        # Terminal cost
        x_err_final = self.X[:, self.N] - self.X_ref[:, self.N]
        obj += ca.mtimes([x_err_final.T, self.Q_terminal, x_err_final])

        self.opti.minimize(obj)

        # Constraints

        # 1. Initial condition constraint
        self.opti.subject_to(self.X[:, 0] == self.X0)

        # 2. Dynamics constraints (using RK4 integration)
        for k in range(self.N):
            x_next = self._rk4_step(self.X[:, k], self.U[:, k], self.dt)
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # 3. Control input constraints
        for k in range(self.N):
            self.opti.subject_to(self.opti.bounded(self.u_min, self.U[:, k], self.u_max))

        # Solver options
        opts = {
            'ipopt.print_level': 0,          # Suppress IPOPT output
            'print_time': 0,                 # Don't print solve time
            'ipopt.max_iter': 200,           # Maximum iterations (increased for aggressive trajectories)
            'ipopt.tol': 1e-6,               # Convergence tolerance
            'ipopt.acceptable_tol': 1e-4,    # Acceptable solution tolerance
            'ipopt.warm_start_init_point': 'yes'  # Use warm start
        }
        self.opti.solver('ipopt', opts)

        # Initialize warm start variables
        self._initialize_warm_start()

    def _rk4_step(self, x: ca.SX, u: ca.SX, dt: float) -> ca.SX:
        """
        Runge-Kutta 4th order integration step

        Integrates dynamics from x(t) to x(t+dt) using control u

        Args:
            x: Current state
            u: Control input
            dt: Time step

        Returns:
            x_next: State at next time step
        """
        k1 = self.f_dynamics(x, u)
        k2 = self.f_dynamics(x + dt/2 * k1, u)
        k3 = self.f_dynamics(x + dt/2 * k2, u)
        k4 = self.f_dynamics(x + dt * k3, u)

        x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return x_next

    def _initialize_warm_start(self):
        """
        Initialize decision variables for warm starting

        Sets initial guess for state and control trajectories.
        This improves convergence speed.
        """
        nx = 12
        nu = 4

        # Initial guess: hover at origin
        x_init = np.zeros((nx, self.N + 1))
        x_init[2, :] = 1.0  # Hover at 1m altitude

        u_init = np.zeros((nu, self.N))
        u_init[0, :] = self.mass * self.g  # Hover thrust

        self.opti.set_initial(self.X, x_init)
        self.opti.set_initial(self.U, u_init)

        # Store for reuse
        self.last_solution_X = x_init
        self.last_solution_U = u_init

    def compute_control(self, current_state: np.ndarray,
                       reference_trajectory: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Solve MPC optimization problem

        Given the current state and desired reference trajectory, solves the
        finite-horizon optimal control problem and returns the first control input.

        Args:
            current_state: Current 12D state vector
            reference_trajectory: Reference states for N+1 time steps (12 x N+1)

        Returns:
            Tuple containing:
                - optimal_control: First control input from solution (4D)
                - solve_time: Time taken to solve in milliseconds
                - success: Whether optimization succeeded
        """
        # Set parameters
        self.opti.set_value(self.X0, current_state)
        self.opti.set_value(self.X_ref, reference_trajectory)

        # Warm start with previous solution (shifted)
        if hasattr(self, 'last_solution_X') and hasattr(self, 'last_solution_U'):
            # Shift previous solution
            X_warm = np.hstack([self.last_solution_X[:, 1:],
                               self.last_solution_X[:, -1:]])
            U_warm = np.hstack([self.last_solution_U[:, 1:],
                               self.last_solution_U[:, -1:]])

            self.opti.set_initial(self.X, X_warm)
            self.opti.set_initial(self.U, U_warm)

        # Solve
        t_start = time.time()

        try:
            sol = self.opti.solve()
            solve_time = (time.time() - t_start) * 1000  # Convert to ms

            # Extract solution
            X_opt = sol.value(self.X)
            U_opt = sol.value(self.U)

            # Store for warm start next iteration
            self.last_solution_X = X_opt
            self.last_solution_U = U_opt

            # Extract first control input
            u_opt = U_opt[:, 0]

            # Update statistics
            self.solve_count += 1
            self.total_solve_time += solve_time

            return u_opt, solve_time, True

        except Exception as e:
            solve_time = (time.time() - t_start) * 1000

            print(f"MPC solve failed: {e}")
            print(f"Current state: {current_state}")

            # Return safe fallback control (hover)
            u_safe = np.array([self.mass * self.g, 0.0, 0.0, 0.0])

            # Update failure statistics
            self.failures += 1

            return u_safe, solve_time, False

    def update_weights(self, Q_new: np.ndarray, R_new: np.ndarray,
                      Q_terminal_new: Optional[np.ndarray] = None):
        """
        Update MPC weight matrices

        This method is called by the RL agent during training to adjust
        the MPC cost function weights.

        Args:
            Q_new: New state cost matrix (12x12 or 12-vector for diagonal)
            R_new: New control cost matrix (4x4 or 4-vector for diagonal)
            Q_terminal_new: New terminal cost matrix (optional)
        """
        # Update weights
        if Q_new.ndim == 1:
            self.Q = np.diag(Q_new)
        else:
            self.Q = Q_new

        if R_new.ndim == 1:
            self.R = np.diag(R_new)
        else:
            self.R = R_new

        if Q_terminal_new is not None:
            if Q_terminal_new.ndim == 1:
                self.Q_terminal = np.diag(Q_terminal_new)
            else:
                self.Q_terminal = Q_terminal_new
        else:
            self.Q_terminal = 10.0 * self.Q

        # Re-setup optimization with new weights
        self.setup_optimization()

    def get_statistics(self) -> dict:
        """
        Get MPC performance statistics

        Returns:
            Dictionary containing solve statistics
        """
        avg_solve_time = (self.total_solve_time / self.solve_count
                         if self.solve_count > 0 else 0)

        return {
            'total_solves': self.solve_count,
            'failures': self.failures,
            'success_rate': ((self.solve_count - self.failures) / self.solve_count * 100
                           if self.solve_count > 0 else 0),
            'avg_solve_time_ms': avg_solve_time,
            'total_solve_time_s': self.total_solve_time / 1000
        }

    def reset_statistics(self):
        """Reset performance statistics"""
        self.solve_count = 0
        self.total_solve_time = 0
        self.failures = 0

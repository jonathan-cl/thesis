from first_exercise.utility import get_survival_prob, get_annuity_payment, simulate, plot_simulation
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import time

start = time.time()

# Model Parameters
beta = 0.96  # time preference
w_0 = 100  # initial wealth
gamma = 5  # risk aversion coefficient
s = 65  # starting age
T = 70  # final period
r = 1.02  # interest rate
p_survival = get_survival_prob(s, T)  # male survival probabilities from 2019

# Utility function; CRRA
def u(c):
    return c**(1-gamma) / (1-gamma)


# Optimize over c
def obj_bellman(c, t, w, an_i, an_payout):
    a = w-c
    a = max(a, grid_w[0])  # Fix this; How to deal with the case w=c -> a=0? Outside of grid
    v_n = interp1d(grid_w, v[t+1][an_i], fill_value='extrapolate')(a * r + an_payout)  # Value function for next period
    return -1 * (u(c) + beta * p_survival[t] * v_n)


n_t = T-s+1  # Number of simulated periods
# Initialize grids
# Wealth
temp = list(range(110))
temp[0] = 0.0001
grid_w = np.array(temp)
# annuity_investment
grid_an = np.arange(0, 101, 5)
# annuity payout
an_payout = get_annuity_payment(grid_an, s, T, r)

# Initialize solution arrays
c = np.nan*np.zeros((n_t, len(grid_an), len(grid_w)))
v = np.nan*np.zeros((n_t, len(grid_an), len(grid_w)))

# Solve
# Last period, consume everything
for an_i in range(len(grid_an)):
    c[-1][an_i] = grid_w
v[-1] = [u(cons) for cons in c[-1]]

# Backwards induction
for t in reversed(range(n_t-1)):
    for an_i, an in enumerate(an_payout):
        for w_i, w in enumerate(grid_w):
            opt = minimize_scalar(obj_bellman, args=(t, w, an_i, an), bounds=(grid_w[0], w), method="bounded")
            c[t, an_i, w_i] = opt.x
            v[t, an_i, w_i] = -obj_bellman(opt.x, t, w, an_i, an)

# Simulate
w_0 = 100
an_investment = 50
sim_c, sim_w = simulate(c, w_0, grid_w, an_investment, an_payout[10], r)
plot_simulation(sim_c, sim_w, s, T, w_0, an_investment)

# Print elapsed time
print(time.time() - start)

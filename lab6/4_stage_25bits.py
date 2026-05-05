

import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
from collections import deque
import random

N      = 4096
T      = 1 / 1e7
f_sig  = 1 / T / N * 13
t      = np.linspace(0, N * T, N, endpoint=False)
Vin_cal    = 0.5 + 0.5 * np.sin(2 * np.pi * f_sig * t)
Vfs    = 1.0

# Fix ONE set of errors for calibration experiments
# (so all algorithms compete on the same problem)
np.random.seed(42)
comparator_offsets_fixed = []
RA_errors_fixed          = []
for _ in range(4):
    comparator_offsets_fixed.append([
        np.random.normal(0, 30e-3),
        np.random.normal(0, 30e-3),
        np.random.normal(0, 30e-3),
        np.random.normal(0, 30e-3),
        np.random.normal(0, 30e-3),
        np.random.normal(0, 30e-3)
    ])
    RA_errors_fixed.append(np.random.normal(0, 30e-3))

def analog_to_thermometer_vec(x, Vfs, comparator_offsets, comparator_knobs):
    """Vectorized: x is shape (N,), returns thermometer shape (N,6)."""
    Vr=Vfs/2
    refs = np.array([-5*Vr,-3*Vr,-Vr,Vr,3*Vr,5*Vr])/8+Vr #(6,)
    offsets = np.array(comparator_offsets) + np.array(comparator_knobs)  # (6,)
    # x[:,None] broadcasts over all 3 comparators at once
    Vin_comp = (x[:, None] - refs[None, :]) + offsets[None, :]  # (N,6)
    return (Vin_comp > 0).astype(int)                             # (N,6)

def thermometer_to_binary_vec(therm):
    """Vectorized: therm shape (N,6), returns codes shape (N,)."""
    # 2.5-bit stage requires 3 bits (0 to 6)
    b2 = np.zeros(len(therm), dtype=int)
    b1 = np.zeros(len(therm), dtype=int)
    b0 = np.zeros(len(therm), dtype=int)

    t5 = therm[:,5]
    t4 = therm[:,4]
    t3 = therm[:,3]
    t2 = therm[:,2]
    t1 = therm[:,1]
    t0 = therm[:,0]

    # Priority encoder masks: checks for the highest '1'
    mask_6 = t5 == 1
    mask_5 = (~mask_6) & (t4 == 1)
    mask_4 = (~(mask_6 | mask_5)) & (t3 == 1)
    mask_3 = (~(mask_6 | mask_5 | mask_4)) & (t2 == 1)
    mask_2 = (~(mask_6 | mask_5 | mask_4 | mask_3)) & (t1 == 1)
    mask_1 = (~(mask_6 | mask_5 | mask_4 | mask_3 | mask_2)) & (t0 == 1)

    # Assign binary values based on the active mask
    # Decimal 6 -> Binary 110
    b2[mask_6] = 1; b1[mask_6] = 1; b0[mask_6] = 0

    # Decimal 5 -> Binary 101
    b2[mask_5] = 1; b1[mask_5] = 0; b0[mask_5] = 1

    # Decimal 4 -> Binary 100
    b2[mask_4] = 1; b1[mask_4] = 0; b0[mask_4] = 0

    # Decimal 3 -> Binary 011
    b2[mask_3] = 0; b1[mask_3] = 1; b0[mask_3] = 1

    # Decimal 2 -> Binary 010
    b2[mask_2] = 0; b1[mask_2] = 1; b0[mask_2] = 0

    # Decimal 1 -> Binary 001
    b2[mask_1] = 0; b1[mask_1] = 0; b0[mask_1] = 1

    return b2 * 4 + b1 * 2 + b0

def stage_vec(Vin, Vfs, comparator_offsets, RA_error,
              comparator_knobs, RA_knob):
    """Vectorized stage: Vin shape (N,)."""
    therm  = analog_to_thermometer_vec(Vin, Vfs,
                                        comparator_offsets,
                                        comparator_knobs)
    codes  = thermometer_to_binary_vec(therm)          # (N,) int 0-6
    Vres   = Vin - codes * Vfs / 8
    Vout   = (1 + RA_error + RA_knob) * 4 * Vres
    return codes, Vout

def ADC_vec(Vin, Vfs, comparator_offsets, RA_errors,
            comparator_knobs, RA_knobs):
    """
    Fully vectorized ADC: Vin shape (N,).
    Returns output codes shape (N,) — replaces the list comprehension.
    ~50-100x faster than the scalar loop version.
    """
    c3, V3 = stage_vec(Vin, Vfs, comparator_offsets[3], RA_errors[3],
                        comparator_knobs[3], RA_knobs[3])
    c2, V2 = stage_vec(V3,  Vfs, comparator_offsets[2], RA_errors[2],
                        comparator_knobs[2], RA_knobs[2])
    c1, V1 = stage_vec(V2,  Vfs, comparator_offsets[1], RA_errors[1],
                        comparator_knobs[1], RA_knobs[1])
    c0, V0 = stage_vec(V1,  Vfs, comparator_offsets[0], RA_errors[0],
                        comparator_knobs[0], RA_knobs[0])
    return c3*64 + c2*16 + c1*4 + c0*1

def compute_SNDR(comparator_knobs, RA_knobs,
                 comp_offsets=comparator_offsets_fixed,
                 ra_errors=RA_errors_fixed):
    Vout = ADC_vec(Vin_cal, Vfs, comp_offsets, ra_errors,   # ← Vin_cal
                   comparator_knobs, RA_knobs).astype(float)
    xf           = scipy.fft.fft(Vout)
    peak_idx     = np.argmax(np.abs(xf[1:N//2])) + 1
    signal_power = np.abs(xf[peak_idx])**2
    noise_power  = np.sum(np.abs(xf[1:N//2])**2) - signal_power
    return 10 * np.log10(signal_power / noise_power)

comparator_offsets = []
RA_errors = []
comparator_knobs = [[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]]
RA_knobs = [0, 0, 0, 0]
for _ in range(4):
  comparator_offsets.append(np.random.normal(0,0,6))
  RA_errors.append(np.random.normal(0,0))

N=4096
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vin = 0.5+0.5*np.sin(2*np.pi*f_sig*t)

Vout = ADC_vec(Vin, 1, comparator_offsets, RA_errors,
                     comparator_knobs, RA_knobs)/512
plt.plot(t, Vin)
plt.plot(t, Vout)
plt.title("Ideal 4-stage pipeline ADC transfer curve")
plt.xlabel(r"$V_{in}\ [V_{fs}]$")
plt.ylabel("Output Code")
plt.grid()
plt.show()

xf = scipy.fft.fft(Vout)
xf_max = max(np.abs(xf))
xf_db = 20*np.log10(np.abs(xf)/xf_max+1e-12)
f = scipy.fft.fftfreq(N, T)[:N//2]
plt.semilogx(f, xf_db[0:N//2])
plt.title(r"PSD of $V_{out}$""\n Ideal 4-stage pipeline ADC")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Relative Power [dB]")
plt.ylim([-200,5])
plt.show()

peak_idx = np.argmax(np.abs(xf[1:N//2])) + 1
signal_power = np.abs(xf[peak_idx])**2

total_power = np.sum((np.abs(xf[1:N//2]))**2)
noise_power = total_power-signal_power

#print(total_power)
#print(signal_power)
#print(noise_power)

SNDR = 10*np.log10(signal_power/noise_power)
print(f"SNDR: {SNDR}")
print(f"ENOB: {(SNDR-1.76)/6.02}")

comparator_offsets = []
RA_errors = []
comparator_knobs = [[0, 0, 0, 0, 0 ,0],
                    [0, 0, 0, 0, 0 ,0],
                    [0, 0, 0, 0, 0 ,0],
                    [0, 0, 0, 0, 0 ,0]]
RA_knobs = [0, 0, 0, 0]
for _ in range(4):
  comparator_offsets.append(np.random.normal(0,30e-3,6))
  RA_errors.append(np.random.normal(0,30e-3))

N = 4096
Vin = np.linspace(0,1,1024)
decimal_codes = ADC_vec(Vin, 1, comparator_offsets, RA_errors,
                     comparator_knobs, RA_knobs)
plt.step(Vin, decimal_codes)
plt.title("4-stage pipeline ADC with errors transfer curve")
plt.xlabel(r"$V_{in}\ [V_{fs}]$")
plt.ylabel("Output Code")
plt.grid()
plt.show()

N=4096
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vin = 0.5+0.5*np.sin(2*np.pi*f_sig*t)

comparator_knobs = [[0, 0, 0, 0, 0 ,0],
                    [0, 0, 0, 0, 0 ,0],
                    [0, 0, 0, 0, 0 ,0],
                    [0, 0, 0, 0, 0 ,0]]
RA_knobs = [0, 0, 0, 0]

SNDRs=[]

for _ in range(100):

  comparator_offsets = []
  RA_errors = []

  for _ in range(4):
    comparator_offsets.append(np.random.normal(0,30e-3,6))
    RA_errors.append(np.random.normal(0,30e-3))

  SNDR = compute_SNDR(comparator_knobs, RA_knobs, comparator_offsets, RA_errors)
  SNDRs.append(SNDR)

plt.hist(SNDRs)
plt.title("Monte Carlo simulation of 4-stage pipeline ADC")
plt.xlabel("SNDR")
plt.axvline(np.mean(SNDRs),color='r')
plt.show()

plt.hist(SNDRs)
plt.title("Monte Carlo simulation of 4-stage pipeline ADC")
plt.xlabel("SNDR")
plt.axvline(np.mean(SNDRs), color='r')
plt.show()
print(f"Average SNDR: {np.mean(SNDRs)}")

"""ML-based calibration section"""

N      = 4096
T      = 1 / 1e7
f_sig  = 1 / T / N * 13
t      = np.linspace(0, N * T, N, endpoint=False)
Vin_cal = 0.5 + 0.5 * np.sin(2 * np.pi * f_sig * t)
Vfs    = 1.0

# ─────────────────────────────────────────────
# KNOB VECTOR HELPERS
# ─────────────────────────────────────────────
# We'll represent knobs as a flat numpy array for optimization:
# indices 0..11  → comparator knobs  (4 stages × 6 comparators)
# indices 12..15 → RA knobs          (4 stages)
NUM_COMP_KNOBS = 4 * 6   # 24
NUM_RA_KNOBS   = 4
KNOB_DIM       = NUM_COMP_KNOBS + NUM_RA_KNOBS  # 28
def flat_to_knobs(theta):
    """Convert flat array (24,) → comparator_knobs, RA_knobs."""
    comp = [list(theta[s*6:(s+1)*6]) for s in range(4)]
    ra   = list(theta[24:28])
    return comp, ra

def knobs_to_flat(comparator_knobs, RA_knobs):
    flat = []
    for s in range(4):
        flat.extend(comparator_knobs[s])
    flat.extend(RA_knobs)
    return np.array(flat, dtype=float)

def eval_theta(theta):
    """Evaluate SNDR for a flat knob vector."""
    comp, ra = flat_to_knobs(theta)
    return compute_SNDR(comp, ra)

# Baseline SNDR (no calibration)
theta_zero    = np.zeros(KNOB_DIM)
sndr_baseline = eval_theta(theta_zero)
print(f"Baseline SNDR (no calibration): {sndr_baseline:.2f} dB")

def lms_calibration(n_iterations=200,
                    mu=0.02,
                    mu_decay=0.990):

    theta      = np.zeros(KNOB_DIM)
    theta_best = np.zeros(KNOB_DIM)
    sndr_best  = eval_theta(theta)
    sndr_hist  = []
    mu_curr    = mu

    zero_offsets    = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4
    zero_errors     = [0.0] * 4
    zero_knobs_comp = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4
    zero_knobs_ra   = [0.0, 0.0, 0.0, 0.0]
    y_ideal = ADC_vec(Vin_cal, Vfs,
                      zero_offsets, zero_errors,
                      zero_knobs_comp, zero_knobs_ra).astype(float)

    fd_delta = 5e-3

    for n in range(n_iterations):
        sndr_curr = eval_theta(theta)
        sndr_hist.append(sndr_curr)

        if sndr_curr > sndr_best:
            sndr_best  = sndr_curr
            theta_best = theta.copy()

        comp_knobs, ra_knobs = flat_to_knobs(theta)

        y_curr = ADC_vec(Vin_cal, Vfs,
                         comparator_offsets_fixed, RA_errors_fixed,
                         comp_knobs, ra_knobs).astype(float)

        e = y_ideal - y_curr

        grad_theta = np.zeros(KNOB_DIM)
        for k in range(KNOB_DIM):
            theta_plus     = theta.copy()
            theta_plus[k] += fd_delta
            comp_p, ra_p   = flat_to_knobs(theta_plus)

            y_plus = ADC_vec(Vin_cal, Vfs,
                             comparator_offsets_fixed, RA_errors_fixed,
                             comp_p, ra_p).astype(float)

            dy_dtheta_k    = (y_plus - y_curr) / fd_delta
            grad_theta[k]  = np.mean(e * dy_dtheta_k)

        # ── Normalize gradient so mu is meaningful regardless of scale
        grad_norm = np.linalg.norm(grad_theta)
        if grad_norm > 1e-10:
            grad_theta_normalized = grad_theta / grad_norm
        else:
            grad_theta_normalized = grad_theta

        theta += mu_curr * grad_theta_normalized   # ← normalized step
        theta  = np.clip(theta, -0.15, 0.15)

        # ── NO ratchet here — LMS must be free to descend the error surface
        # The ratchet was resetting theta to zero every iteration

        mu_curr *= mu_decay

        if n % 20 == 0:
            nonzero  = np.count_nonzero(grad_theta)
            rms_e    = np.sqrt(np.mean(e**2))
            theta_norm = np.linalg.norm(theta)
            print(f"[LMS] iter {n:4d}  SNDR={sndr_curr:.2f} dB  "
                  f"best={sndr_best:.2f} dB  "
                  f"nonzero={nonzero}/{KNOB_DIM}  "
                  f"RMS_e={rms_e:.3f}  "
                  f"|θ|={theta_norm:.4f}  "    # ← if this stays 0, theta isn't moving
                  f"μ={mu_curr:.5f}")

    return theta_best, sndr_hist

theta_gd, sndr_gd = lms_calibration(n_iterations=200)
print(f"[LMS] Final SNDR: {sndr_gd[-1]:.2f} dB")

def cps_calibration(n_iterations=200, M=8, I_trans=20, gamma=1.0):
    """
    Centroid Pull Search (CPS) as described in the lab background [1].

    M       : number of centroids per parameter
    I_trans : transient (exploration) phase length
    gamma   : decay constant for step size after transient
    """
    # Spread centroids symmetrically around zero in [-0.1, 0.1]
    centroids = np.linspace(-0.1, 0.1, M)

    theta     = np.zeros(KNOB_DIM)
    sndr_hist = []

    for n in range(n_iterations):
        # Adaptive step size η(n)
        if n < I_trans:
            eta = 1.0
        else:
            eta = 1.0 / (gamma * (n - I_trans + 1))
        eta = np.clip(eta, 0.01, 1.0)   # prevent vanishing step

        sndr_curr = eval_theta(theta)
        sndr_hist.append(sndr_curr)

        # For each parameter dimension, find best centroid pull
        for k in range(KNOB_DIM):
            best_sndr  = -np.inf
            best_theta_k = theta[k]

            for m in range(M):
                theta_cand    = theta.copy()
                # Candidate: pull current estimate toward centroid c_m
                theta_cand[k] = theta[k] + eta * (centroids[m] - theta[k])
                sndr_cand     = eval_theta(theta_cand)

                if sndr_cand > best_sndr:
                    best_sndr    = sndr_cand
                    best_theta_k = theta_cand[k]

            theta[k] = best_theta_k

        if n % 20 == 0:
            print(f"[CPS] iter {n:4d}  SNDR = {sndr_curr:.2f} dB  eta={eta:.4f}")

    return theta, sndr_hist

theta_cps, sndr_cps = cps_calibration (n_iterations=200)
print(f"[CPS] Final SNDR: {sndr_cps[-1]:.2f} dB")

def build_neural_module(input_dim=KNOB_DIM + 1, output_dim=KNOB_DIM):
    """
    Lightweight neural network that proposes a Δθ correction
    given (E_G, θ_hat) as input.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(output_dim, activation='tanh')
    ])
    return model

def ai_loops_pts_calibration(n_iterations=200, M=6, N_s=3,
                              I_trans=30, gamma=1.0, lr_nn=1e-3):
    """
    AI-Loops with Parallel Thompson Sampling (PTS) [1].

    M         : total number of neural modules (arms)
    N_s       : subset size evaluated each iteration (N_s < M)
    I_trans   : full-evaluation transient phase
    gamma     : step-size decay constant
    lr_nn     : learning rate for neural module updates
    """
    # ── Build M neural modules ──────────────────────────────────────
    modules    = [build_neural_module() for _ in range(M)]
    optimizers = [tf.keras.optimizers.Adam(lr_nn) for _ in range(M)]

    # ── PTS Beta posteriors: α_m, β_m initialized to (1,1) ──────────
    alpha = np.ones(M)
    beta  = np.ones(M)

    theta     = np.zeros(KNOB_DIM, dtype=np.float32)
    sndr_hist = []

    # ── Transient phase: evaluate ALL M modules to build priors ─────
    hit_count = np.zeros(M, dtype=int)

    for n in range(n_iterations):
        sndr_curr = float(eval_theta(theta))
        sndr_hist.append(sndr_curr)

        # Step size η(n)
        if n < I_trans:
            eta = 1.0
        else:
            eta = 1.0 / (gamma * (n - I_trans + 1))
        eta = float(np.clip(eta, 0.01, 1.0))

        # Normalized error signal fed to networks
        E_G = sndr_baseline - sndr_curr   # positive when worse than baseline
        nn_input = np.concatenate([[E_G], theta]).astype(np.float32)
        nn_input_t = tf.constant(nn_input[np.newaxis, :])

        # ── Subset selection via Thompson Sampling ───────────────────
        if n < I_trans:
            subset = list(range(M))          # evaluate all during transient
        else:
            samples = np.array([
                np.random.beta(alpha[m], beta[m]) for m in range(M)
            ])
            subset = list(np.argsort(samples)[-N_s:])   # top-N_s

        # ── Evaluate each candidate in subset ────────────────────────
        best_sndr   = -np.inf
        best_theta  = theta.copy()
        best_m      = subset[0]

        for m in subset:
            delta_theta = modules[m](nn_input_t).numpy().flatten()
            theta_cand  = theta + eta * delta_theta

            sndr_cand = eval_theta(theta_cand)
            if sndr_cand > best_sndr:
                best_sndr  = sndr_cand
                best_theta = theta_cand
                best_m     = m

        # ── Update parameter vector ───────────────────────────────────
        theta = best_theta.astype(np.float32)

        # ── PTS Posterior update (Beta-Bernoulli) ─────────────────────
        for m in subset:
            reward    = 1 if m == best_m else 0
            alpha[m] += reward
            beta[m]  += (1 - reward)
            if n < I_trans and reward == 1:
                hit_count[m] += 1

        # ── Train winning module to reinforce its update ──────────────
        # Supervised: the winning Δθ should point toward (best_theta - old_theta)
        target = tf.constant((best_theta - theta)[np.newaxis, :],
                             dtype=tf.float32)
        with tf.GradientTape() as tape:
            pred = modules[best_m](nn_input_t)
            loss = tf.reduce_mean(tf.square(pred - target))
        grads = tape.gradient(loss, modules[best_m].trainable_variables)
        optimizers[best_m].apply_gradients(
            zip(grads, modules[best_m].trainable_variables))

        if n % 20 == 0:
            print(f"[AI-PTS] iter {n:4d}  SNDR = {sndr_curr:.2f} dB  "
                  f"best_m={best_m}  η={eta:.4f}")

    return theta, sndr_hist

theta_ai_loops_PTS, sndr_ai_loops_PTS = ai_loops_pts_calibration(n_iterations=200)
print(f"[AI-PTS] Final SNDR: {sndr_ai_loops_PTS[-1]:.2f} dB")

# ── Replay Buffer ────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns = zip(*batch)
        return (np.array(s),  np.array(a),
                np.array(r),  np.array(ns))

    def __len__(self):
        return len(self.buffer)

# ── Q-Network ────────────────────────────────────────────────────────────────
def build_q_network(state_dim, n_actions):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu',
                              input_shape=(state_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(n_actions)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='mse')
    return model

def rl_dqn_calibration(n_iterations=200,
                       step_size=8e-3,
                       epsilon_start=1.0,
                       epsilon_end=0.05,
                       epsilon_decay=0.95,        # verify this is what's called
                       gamma_rl=0.95,
                       batch_size=32,
                       update_target_every=5,
                       exploit_from=100):         # ← NEW: switch to exploit at iter 100
    """
    Two-phase DQN:
    Phase 1 (0 → exploit_from):    Normal ε-greedy, build replay buffer,
                                    train Q-network, track best theta.
    Phase 2 (exploit_from → end):  Reset theta to best found, ε=0,
                                    pure exploitation — Q-network refines
                                    from the best known starting point.
    """
    STATE_DIM  = KNOB_DIM
    N_ACTIONS  = KNOB_DIM * 2
    q_net      = build_q_network(STATE_DIM, N_ACTIONS)
    q_target   = build_q_network(STATE_DIM, N_ACTIONS)
    q_target.set_weights(q_net.get_weights())
    optimizer  = tf.keras.optimizers.Adam(1e-3)
    buffer     = ReplayBuffer(capacity=5000)

    theta      = np.zeros(KNOB_DIM)
    epsilon    = epsilon_start
    sndr_hist  = []
    sndr_prev  = eval_theta(theta)
    theta_best = theta.copy()
    sndr_best  = sndr_prev
    phase      = 1

    for n in range(n_iterations):
        # ── Phase transition ──────────────────────────────────────
        if n == exploit_from and phase == 1:
            # Reset to best found, switch to pure exploitation
            theta     = theta_best.copy()
            sndr_prev = eval_theta(theta)
            epsilon   = 0.0        # ← pure greedy from here
            phase     = 2
            print(f"[DQN] ── Phase 2: exploit from best={sndr_best:.2f} dB ──")

        sndr_hist.append(sndr_best)   # plot running best
        state = theta.copy()

        # ── ε-greedy ──────────────────────────────────────────────
        if np.random.rand() < epsilon:
            action = np.random.randint(0, N_ACTIONS)
        else:
            q_vals = q_net(state[np.newaxis, :].astype(np.float32))
            action = int(np.argmax(q_vals.numpy()))

        knob_idx  = action // 2
        direction = +1 if (action % 2 == 0) else -1

        theta_new           = theta.copy()
        theta_new[knob_idx] = np.clip(
            theta_new[knob_idx] + direction * step_size, -0.15, 0.15)

        sndr_new = eval_theta(theta_new)

        # ── In phase 2: only accept improvements ──────────────────
        if phase == 2 and sndr_new < sndr_prev:
            # Reject the action — don't move
            theta_new = theta.copy()
            sndr_new  = sndr_prev

        reward = np.tanh((sndr_new - sndr_prev) / 5.0)

        buffer.push(state, action, reward, theta_new)
        theta     = theta_new
        sndr_prev = sndr_new

        if sndr_new > sndr_best:
            sndr_best  = sndr_new
            theta_best = theta_new.copy()

        # ── Train Q-network ───────────────────────────────────────
        if len(buffer) >= batch_size:
            states_b, actions_b, rewards_b, next_states_b = \
                buffer.sample(batch_size)

            states_t      = tf.constant(states_b,      dtype=tf.float32)
            next_states_t = tf.constant(next_states_b, dtype=tf.float32)
            rewards_t     = tf.constant(rewards_b,     dtype=tf.float32)

            q_next  = q_target(next_states_t).numpy()
            targets = rewards_t.numpy() + gamma_rl * np.max(q_next, axis=1)

            with tf.GradientTape() as tape:
                q_pred    = q_net(states_t)
                q_pred_np = q_pred.numpy().copy()
                for b in range(batch_size):
                    q_pred_np[b, actions_b[b]] = targets[b]
                target_t = tf.constant(q_pred_np, dtype=tf.float32)
                loss     = tf.reduce_mean(tf.square(q_pred - target_t))

            grads = tape.gradient(loss, q_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_net.trainable_variables))

        if n % update_target_every == 0:
            q_target.set_weights(q_net.get_weights())

        # ── Only decay epsilon in phase 1 ─────────────────────────
        if phase == 1:
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if n % 20 == 0:
            print(f"[DQN] iter {n:4d}  current={sndr_prev:.2f} dB  "
                  f"best={sndr_best:.2f} dB  ε={epsilon:.3f}  phase={phase}")

    return theta_best, sndr_hist

theta_dqn, sndr_dqn = rl_dqn_calibration(
    n_iterations=200,
    epsilon_decay=0.95,
    exploit_from=100
)
print(f"[DQN] Final best SNDR: {eval_theta(theta_dqn):.2f} dB")

# ═══════════════════════════════════════════════════════════════════
# AGENTIC AI CALIBRATION
# ═══════════════════════════════════════════════════════════════════
#
# Architecture: 3 specialized agents
# ─────────────────────────────────
#  Agent 0 — "Comparator Agent"
#             Owns comparator knobs for ALL stages (θ[0:12])
#             Strategy: CPS-style centroid search
#             Communicates: reports SNDR delta to coordinator
#
#  Agent 1 — "Gain Agent"
#             Owns RA (residue amplifier) knobs for ALL stages (θ[12:16])
#             Strategy: gradient-based (SPSA)
#             Communicates: reports confidence score to coordinator
#
#  Agent 2 — "Coordinator Agent"
#             Owns NO knobs directly
#             Reads reports from Agent 0 and Agent 1
#             Decides: which agent's update to TRUST each iteration
#             Strategy: weighted voting based on recent reward history
#             Also detects stagnation and triggers "shake" resets
#
# Why this decomposition?
# ───────────────────────
# Comparator offsets and gain mismatches have DIFFERENT error surfaces:
# - Offsets create threshold shifts → discrete jumps → CPS handles well
# - Gain errors create smooth scaling → gradient methods handle well
# Separating them lets each agent use the best tool for its subproblem.
# The coordinator prevents agents from fighting each other.
#
# ═══════════════════════════════════════════════════════════════════

class ComparatorAgent:
    """
    Agent 0: Owns θ[0:12] (comparator knobs, 4 stages × 3 comparators).
    Uses Centroid Pull Search to find offset-cancelling values.
    """
    def __init__(self, n_centroids=8, step_init=1.0,
                 I_trans=15, gamma=1.2):
        self.idx       = slice(0, 12)          # which knob indices we own
        self.M         = n_centroids
        self.centroids = np.linspace(-0.12, 0.12, n_centroids)
        self.step_init = step_init
        self.I_trans   = I_trans
        self.gamma     = gamma
        self.iteration = 0
        self.reward_history = deque(maxlen=20)  # recent SNDR deltas

    def propose_update(self, theta_current, sndr_current):
        """
        Proposes a new sub-vector for the comparator knobs.
        Returns (proposed_theta_full, confidence).
        """
        n   = self.iteration
        eta = (1.0 if n < self.I_trans
               else 1.0 / (self.gamma * (n - self.I_trans + 1)))
        eta = np.clip(eta, 0.008, 1.0)

        theta_prop = theta_current.copy()
        my_slice   = theta_current[self.idx].copy()

        # CPS: for each of the 12 comparator knobs, find best centroid pull
        for k in range(12):
            best_sndr = -np.inf
            best_val  = my_slice[k]
            for c in self.centroids:
                candidate_val = my_slice[k] + eta * (c - my_slice[k])
                theta_test    = theta_current.copy()
                theta_test[k] = candidate_val
                s = eval_theta(theta_test)
                if s > best_sndr:
                    best_sndr = s
                    best_val  = candidate_val
            my_slice[k] = best_val

        theta_prop[self.idx] = my_slice

        # Confidence = mean recent reward (how useful I've been lately)
        confidence = (np.mean(self.reward_history)
                      if len(self.reward_history) > 0 else 0.5)
        return theta_prop, float(confidence)

    def receive_reward(self, sndr_delta):
        """Coordinator calls this to give feedback."""
        self.reward_history.append(sndr_delta)
        self.iteration += 1


class GainAgent:
    """
    Agent 1: Owns θ[12:16] (RA knobs, 4 stages).
    Uses SPSA gradient ascent to cancel gain mismatches.
    """
    def __init__(self, lr=0.04, lr_decay=0.993,
                 delta=0.008, momentum=0.8):
        self.idx            = slice(12, 16)
        self.lr             = lr
        self.lr_decay       = lr_decay
        self.delta          = delta
        self.momentum       = momentum
        self.velocity       = np.zeros(4)
        self.iteration      = 0
        self.reward_history = deque(maxlen=20)

    def propose_update(self, theta_current, sndr_current):
        """
        Proposes updated RA knobs via SPSA gradient step.
        Returns (proposed_theta_full, confidence).
        """
        theta_prop = theta_current.copy()
        my_vals    = theta_current[self.idx].copy()

        # SPSA on just the 4 RA knobs (only 2 evals needed)
        Delta     = (2 * (np.random.rand(4) > 0.5).astype(float) - 1)
        theta_p   = theta_current.copy()
        theta_m   = theta_current.copy()
        theta_p[self.idx] = my_vals + self.delta * Delta
        theta_m[self.idx] = my_vals - self.delta * Delta

        sndr_p = eval_theta(theta_p)
        sndr_m = eval_theta(theta_m)
        grad   = (sndr_p - sndr_m) / (2 * self.delta * Delta)

        # Momentum update
        self.velocity = (self.momentum * self.velocity
                         + (1 - self.momentum) * grad)
        n          = self.iteration + 1
        v_corrected = self.velocity / (1 - self.momentum ** n)

        my_vals += self.lr * v_corrected
        my_vals  = np.clip(my_vals, -0.15, 0.15)
        theta_prop[self.idx] = my_vals

        confidence = (np.mean(self.reward_history)
                      if len(self.reward_history) > 0 else 0.5)
        self.lr *= self.lr_decay
        return theta_prop, float(confidence)

    def receive_reward(self, sndr_delta):
        self.reward_history.append(sndr_delta)
        self.iteration += 1


class CoordinatorAgent:
    """
    Agent 2: The meta-agent. Owns no knobs directly.

    Responsibilities:
    ─────────────────
    1. Solicits proposals from Agent 0 and Agent 1
    2. Evaluates each proposal independently
    3. Assigns rewards back to each agent
    4. Selects winning proposal (or merges both if both improve SNDR)
    5. Detects stagnation (SNDR not improving for N steps) and
       triggers a random "shake" to escape local minima

    Decision policy:
    ─────────────────
    - If BOTH proposals improve SNDR: take the better one
    - If ONE improves: take it, penalize the other
    - If NEITHER improves: keep current θ, trigger shake if stagnant
    - "Merge" mode: occasionally combine both proposals if they
      affect non-overlapping parameter subsets (they always do here,
      since Agent 0 owns idx 0:12 and Agent 1 owns idx 12:16)
    """
    def __init__(self, stagnation_window=30, shake_scale=0.01):
        self.stagnation_window = stagnation_window
        self.shake_scale       = shake_scale
        self.sndr_history      = deque(maxlen=stagnation_window)
        self.merge_count       = 0
        self.shake_count       = 0

    def is_stagnant(self):
        if len(self.sndr_history) < self.stagnation_window:
            return False
        window = list(self.sndr_history)
        return (max(window) - min(window)) < 0.3   # less than 0.3 dB improvement

    def decide(self, theta_current, sndr_current,
               proposal_0, conf_0,
               proposal_1, conf_1):
        """
        Core decision logic.
        Returns (theta_next, decision_label).
        """
        sndr_0 = eval_theta(proposal_0)
        sndr_1 = eval_theta(proposal_1)

        improved_0 = sndr_0 > sndr_current
        improved_1 = sndr_1 > sndr_current

        if improved_0 and improved_1:
            # ── MERGE: since agents own non-overlapping slices,
            #    we can safely combine both proposals ────────────
            theta_merged      = theta_current.copy()
            theta_merged[:12] = proposal_0[:12]   # Agent 0's comparator update
            theta_merged[12:] = proposal_1[12:]   # Agent 1's gain update
            sndr_merged       = eval_theta(theta_merged)

            self.merge_count += 1
            reward_0 = sndr_merged - sndr_current
            reward_1 = sndr_merged - sndr_current
            return theta_merged, sndr_merged, reward_0, reward_1, "MERGE"

        elif improved_0 and not improved_1:
            reward_0 = sndr_0 - sndr_current
            reward_1 = sndr_1 - sndr_current        # negative
            return proposal_0, sndr_0, reward_0, reward_1, "AGENT_0"

        elif improved_1 and not improved_0:
            reward_0 = sndr_0 - sndr_current        # negative
            reward_1 = sndr_1 - sndr_current
            return proposal_1, sndr_1, reward_0, reward_1, "AGENT_1"

        else:
            # Neither improved — check for stagnation
            if self.is_stagnant():
                # "Shake": add small random perturbation to escape local min
                shake = np.random.randn(KNOB_DIM) * self.shake_scale
                theta_shaken = np.clip(theta_current + shake, -0.15, 0.15)
                self.shake_count += 1
                reward_0 = 0.0
                reward_1 = 0.0
                return theta_shaken, eval_theta(theta_shaken), \
                       reward_0, reward_1, "SHAKE"
            else:
                reward_0 = sndr_0 - sndr_current
                reward_1 = sndr_1 - sndr_current
                return theta_current, sndr_current, reward_0, reward_1, "HOLD"


def agentic_ai_calibration(n_iterations=200):
    """
    Full Agentic AI calibration loop.

    3 agents running autonomously with message passing:
      Agent 0 (ComparatorAgent) ─┐
                                  ├─► CoordinatorAgent decides θ(n+1)
      Agent 1 (GainAgent)       ─┘

    Communication protocol each iteration:
    ───────────────────────────────────────
    1. Coordinator broadcasts current (θ, SNDR) to both agents
    2. Each agent independently proposes a new θ + confidence score
    3. Coordinator evaluates proposals and picks winner / merges
    4. Coordinator sends reward signal (SNDR delta) back to each agent
    5. Each agent updates its internal state
    """
    agent_comp  = ComparatorAgent(n_centroids=8, I_trans=15, gamma=1.2)
    agent_gain  = GainAgent(lr=0.04, lr_decay=0.996, momentum=0.8)
    coordinator = CoordinatorAgent(stagnation_window=100, shake_scale=0.001)

    theta     = np.zeros(KNOB_DIM)
    sndr_curr = eval_theta(theta)
    sndr_hist = [sndr_curr]

    print(f"[AGENTIC] Initial SNDR: {sndr_curr:.2f} dB")

    for n in range(n_iterations):
        # ── Step 1: Coordinator broadcasts state ──────────────────
        # (agents receive theta_current, sndr_current implicitly
        #  via their propose_update call)

        # ── Step 2: Agents independently propose updates ──────────
        proposal_0, conf_0 = agent_comp.propose_update(theta, sndr_curr)
        proposal_1, conf_1 = agent_gain.propose_update(theta, sndr_curr)

        # ── Step 3: Coordinator decides ───────────────────────────
        theta_next, sndr_next, r0, r1, decision = coordinator.decide(
            theta, sndr_curr,
            proposal_0, conf_0,
            proposal_1, conf_1
        )

        # ── Step 4: Coordinator sends rewards ─────────────────────
        agent_comp.receive_reward(r0)
        agent_gain.receive_reward(r1)

        # ── Step 5: Update state ───────────────────────────────────
        theta     = theta_next
        sndr_curr = sndr_next
        coordinator.sndr_history.append(sndr_curr)
        sndr_hist.append(sndr_curr)

        if n % 20 == 0:
            print(f"[AGENTIC] iter {n:4d}  SNDR={sndr_curr:.2f} dB  "
                  f"decision={decision:8s}  "
                  f"merges={coordinator.merge_count}  "
                  f"shakes={coordinator.shake_count}")

    print(f"\n[AGENTIC] Final SNDR     : {sndr_hist[-1]:.2f} dB")
    print(f"[AGENTIC] Total merges   : {coordinator.merge_count}")
    print(f"[AGENTIC] Total shakes   : {coordinator.shake_count}")
    return theta, sndr_hist

theta_agentic, sndr_agentic = agentic_ai_calibration(n_iterations=200)
print(f"[AGENTIC] Final SNDR: {sndr_agentic[-1]:.2f} dB")

# ═══════════════════════════════════════════════════════════════════
# COMPARISON PLOT — all 4 algorithms vs. iteration count
# Required by lab: "Plot the SNDR as a function of the number of
# iterations of the algorithms and show all the algorithm plots
# together for comparison purposes."
# ═══════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(sndr_gd, label='Gradient Descent (LMS)',
        color='royalblue', linewidth=2,   linestyle='-')
ax.plot(sndr_cps,           label='Multi-Centroid CPS + PTS',
        color='darkorange', linewidth=2,   linestyle='-')
ax.plot(sndr_dqn,           label='Reinforcement Learning (DQN)',
        color='green',      linewidth=2,   linestyle='-')
ax.plot(sndr_agentic,       label='Agentic AI (3 agents)',
        color='red',        linewidth=2,   linestyle='-')

# Baseline reference line (no calibration)
ax.axhline(y=sndr_baseline,
           color='black', linestyle='--', linewidth=1.5,
           label=f'Baseline (no cal): {sndr_baseline:.1f} dB')

ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('SNDR [dB]',  fontsize=13)
ax.set_title('Calibration Algorithm Comparison\n'
             '4-Stage Pipeline ADC (2.5 bits/stage, σ=30mV offsets, σ=30mV/V gain)',
             fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.4)
ax.set_xlim([0, 200])

# Annotate final SNDR values on the right edge
final_values = {
    'GD':      sndr_gd[-1],
    'CPS+PTS': sndr_cps[-1],
    'DQN':     sndr_dqn[-1],
    'Agentic': sndr_agentic[-1],
}
colors = ['royalblue', 'darkorange', 'green', 'red']
for (label, val), color in zip(final_values.items(), colors):
    ax.annotate(f'{val:.1f} dB',
                xy=(200, val),
                xytext=(203, val),
                color=color,
                fontsize=9,
                va='center')

plt.tight_layout()
plt.savefig('calibration_comparison.png', dpi=150)
plt.show()

# ── Summary table ──────────────────────────────────────────────────
print("\n" + "="*55)
print(f"{'Algorithm':<25} {'Final SNDR':>12}  {'vs Baseline':>12}")
print("="*55)
for (label, val), color in zip(final_values.items(), colors):
    delta = val - sndr_baseline
    print(f"{label:<25} {val:>10.2f} dB  {delta:>+10.2f} dB")
print("="*55)
print(f"{'Baseline (no cal)':<25} {sndr_baseline:>10.2f} dB  {'':>12}")
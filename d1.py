import numpy as np
import sympy
import os, json
import matplotlib.pyplot as plt

sym_u = sympy.Symbol("u")
sym_v = sympy.Symbol("v")

# coefficients
a = -1
b = 1
c = 1
d = 3

sym_W = (
    ((sym_u - a) ** 2 + sym_v ** 2)
    * ((sym_u - b) ** 2 + (sym_v - c) ** 2)
    * ((sym_u - b) ** 2 + (sym_v + c) ** 2)
    * ((sym_u - d) ** 2 + sym_v ** 2)
)
W = sympy.lambdify((sym_u, sym_v), sym_W, "numpy")

DuW = sympy.lambdify((sym_u, sym_v), sym_W.diff(sym_u), "numpy")
DvW = sympy.lambdify((sym_u, sym_v), sym_W.diff(sym_v), "numpy")

# discretization
strip_width = 100
h = mesh_size = 0.001
outside_left_bdy_pt = 0 - mesh_size
outside_right_bdy_pt = strip_width + 2 * mesh_size

# t_-1 < t_0 < t_1 < ... < t_n-1 < t_n < t_n+1
# t_-1 and t_n+1 are used for calculating derivatives at t_0 and t_n.
t_mesh = np.arange(outside_left_bdy_pt, outside_right_bdy_pt, mesh_size)

SAVE_DIR = "save_dir_100"
os.makedirs(os.path.join(SAVE_DIR, "sol_plots"), exist_ok=True)


def F(uu, vv, hh=h):
    # u is fixed to be 1.
    dv = (vv[2:] - vv[:-2]) / (2 * hh)
    return np.sum((dv[1:-1] ** 2) / 2 + W(uu[2:-2], vv[2:-2])) * hh


# load initial condition or restart calculation
if os.path.isfile(f"{SAVE_DIR}/t_mesh.npy"):
    t_mesh = np.load(open(f"{SAVE_DIR}/t_mesh.npy", "rb"))
    v = np.load(open(f"{SAVE_DIR}/v_mesh.npy", "rb"))
    u = b * np.ones(shape=v.shape)
    step_lines = json.load(open(f"{SAVE_DIR}/steps.json"))
    step = step_lines[0]
    lr = step_lines[1]
    plot_initial = False
else:
    os.makedirs(f"{SAVE_DIR}", exist_ok=True)
    u = b * np.ones(shape=len(t_mesh))
    v = np.linspace(0, 0.95 * c, num=len(t_mesh))
    v[0] = v[1] = 0
    v[v < 0] = 0
    v[-1] = v[-2] = v[-3]
    step = 0
    lr = 0.1
    plot_initial = True

# set initial momenta of the ADAM algorithm
pre_m_1 = pre_m_2 = pre_n = 0
EPSILON = 1e-10
total_delta = 1
residual = -1
previous = F(u, v, h)
now_minimum = previous
eq_v = 1

# if the variable now_minimum stops decreasing,
# we should downscale the learning rate.
energy_stop_decreasing_times = 0


# plot of initial conditions
if plot_initial:
    new_t = t_mesh[1::10]
    new_v = v[1::10]
    new_h = (new_t[1:] - new_t[:-1]).max()
    new_u = b * np.ones(shape=len(new_t))
    N = int(len(new_t))
    t_slice = slice(0, N)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(211)
    ax1.plot(new_t[t_slice], new_u[t_slice], label="u")
    ax1.legend()
    ax2 = fig.add_subplot(212)
    ax2.plot(new_t[t_slice], new_v[t_slice], label="v")
    ax2.legend()
    ax1.set_title(f"the u,v-plots at step={step}")
    plt.savefig(os.path.join(SAVE_DIR, "sol_plots", f"plot_{step:08d}.png"))
    plt.close(fig)


while (
    step < 10000050 and np.max(np.abs(eq_v)) > h and lr > 1e-10
):  # and total_delta > 1e-10:
    step += 1
    if energy_stop_decreasing_times > 500:
        lr *= 0.99
        energy_stop_decreasing_times = 0
    if step % 100000 == 0:
        lr *= 1.1

    mu = 0.1 * (1.5 + 1.5 * np.exp(-step / 1000))
    nu = 0.1 * (1.5 + 1.5 * np.exp(-step / 1000))

    dv = (v[2:] - v[:-2]) / (2 * h)
    pFpv = (dv[:-2] - dv[2:]) / 2 + DvW(u[2:-2], v[2:-2]) * h
    m_2 = mu * pre_m_2 + (1 - mu) * pFpv

    this_n = nu * pre_n + (1 - nu) * (np.sum(pFpv ** 2))

    if step < 20:
        m_2 /= 1 - mu ** step
        this_n /= 1 - nu ** step

    np.random.seed(3345678)
    rand_coefs = 0.5 + np.random.random(len(m_2)) * 0.5
    rand_coefs[rand_coefs < 0] = 0

    v_delta = rand_coefs * lr * m_2 / (np.sqrt(this_n) + EPSILON)

    total_delta = np.sum(np.abs(v_delta))

    v[2:-2] -= v_delta

    v[v < 0] = 0
    over_1_pt_inds = np.asarray(np.where(v > 1))
    over_1_pt_inds = over_1_pt_inds[
        (over_1_pt_inds > 0) & (over_1_pt_inds) < len(v.shape)
    ]
    v[v > 1] = 1
    v[over_1_pt_inds] = (
        v[over_1_pt_inds - 1] + v[over_1_pt_inds] + v[over_1_pt_inds + 1]
    ) / 3
    m_2[v[2:-2] < 0] = 0
    m_2[v[2:-2] > 1] = 0

    v[0] = v[1] = 0
    v[-1] = v[-2] = v[-3]

    pre_m_2 = m_2

    energy = F(u, v, h)
    if energy >= now_minimum:
        energy_stop_decreasing_times += 1
    else:
        energy_stop_decreasing_times = 0
        now_minimum = energy
    residual = energy - previous
    previous = energy

    if step % 50 == 0:
        print(  #'\r',
            f"{step:08d} > lr: {lr:.10f}; energy: {energy:14.10f}; ",
            f"res: {residual/energy*100:+2.1f}%; ",
            f"delta: {total_delta:13.10f}; ",
            f"v: {v.min()}, {v.max():.10f}",
            end="\r",
            flush=True,
        )

    if step % 1000 == 0:

        with open(f"{SAVE_DIR}/t_mesh.npy", "wb") as fp:
            np.save(fp, t_mesh)
        with open(f"{SAVE_DIR}/u_mesh.npy", "wb") as fp:
            np.save(fp, u)
        with open(f"{SAVE_DIR}/v_mesh.npy", "wb") as fp:
            np.save(fp, v)

        step_lines = [step, lr]
        with open(f"{SAVE_DIR}/steps.json", "w") as f:
            json.dump(step_lines, f, ensure_ascii=False, indent=4)

    if step % 10000 == 0:
        new_t = t_mesh[1::10]
        new_v = v[1::10]
        new_h = (new_t[1:] - new_t[:-1]).max()
        new_u = b * np.ones(shape=len(new_t))
        eq_v = (new_v[2:] + new_v[:-2] - 2 * new_v[1:-1]) / new_h ** 2 - DvW(
            new_u[1:-1], new_v[1:-1]
        )
        # print('-'*142)
        message = (
            f" {step:08d} > lr: {lr:.10f} /"
            + f" energy: {energy:14.10f} /"
            + f" eq. error: {eq_v.max():14.10f} {eq_v.min():14.10f}"
        )
        message_length = len(message)
        print("[" + message + " " + "-" * (142 - message_length) + "]", end="\n")
        log_dict = dict(
            step=step,
            lr=lr,
            energy=energy,
            accuracy=[eq_v.max(), eq_v.min()],
        )
        with open(f"{SAVE_DIR}/solve_equation.log", "a") as fp:
            json.dump(log_dict, fp, ensure_ascii=False)
            fp.write("\n")

        N = int(len(new_t))
        t_slice = slice(0, N)
        fig = plt.figure(figsize=(12, 6))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0.25)
        ax1 = fig.add_subplot(211)
        ax1.plot(new_t[t_slice], new_u[t_slice], label="u")
        ax1.legend()
        ax2 = fig.add_subplot(212)
        ax2.plot(new_t[t_slice], new_v[t_slice], label="v")
        ax2.legend()
        ax1.set_title(f"the u,v-plots at step={step}")
        ax2.set_title(f"v_xx - DvW(u,v) lies in ({eq_v.min()}, {eq_v.max()})")
        plt.savefig(os.path.join(SAVE_DIR, "sol_plots", f"plot_{step:08d}.png"))
        # fig.clf()
        plt.close(fig)

print("")
print(f"Stop at step = {step} and delta = {total_delta}.")

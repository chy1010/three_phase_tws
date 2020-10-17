import numpy as np
from numpy.core.fromnumeric import size
import sympy, time
import matplotlib.pyplot as plt
import os, sys, json

sym_u = sympy.Symbol("u")
sym_v = sympy.Symbol("v")

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


# set boundary & meshsize
LOAD_1D_DIR = "save_dir_100"
t_mesh = np.load(open(f"{LOAD_1D_DIR}/t_mesh.npy", "rb"))
phi_v = np.load(open(f"{LOAD_1D_DIR}/v_mesh.npy", "rb"))

t_mesh = t_mesh[1::10]
phi_v = phi_v[1::10]

phi_v[-1] = phi_v[-2] = phi_v[-3]

x_mesh = np.arange(-20, 20 + 0.01, 0.01)
h = k = 0.01
CONSTRAINT = 1650

t_mesh = np.concatenate([np.asarray([-h]), t_mesh], axis=0)
phi_u = b * np.ones(shape=t_mesh.shape)
phi_v = np.concatenate([np.asarray([0]), phi_v], axis=0)


SAVE_DIR = "save_dir_2d"
os.makedirs(f"{SAVE_DIR}/UV_y_sec_plots", exist_ok=True)

if os.path.isfile(f"{SAVE_DIR}/t_mesh.npy"):
    U = np.load(open(f"{SAVE_DIR}/U_mesh.npy", "rb"))
    V = np.load(open(f"{SAVE_DIR}/V_mesh.npy", "rb"))
    step_lines = json.load(open(f"{SAVE_DIR}/steps.json"))
    step = step_lines[0]
    lr = step_lines[1]
    plot_initial = False

else:
    ii = np.linspace(0, 1, num=int(len(x_mesh) / 2) - 1800)
    jj = np.zeros(shape=1600)
    ii = np.concatenate(
        [jj, ii, np.ones(shape=len(x_mesh) - len(ii) - len(jj))], axis=0
    )
    del jj
    U = a * (1 - ii)[:, np.newaxis] + ii[:, np.newaxis] * phi_u[np.newaxis, :]
    V = ii[:, np.newaxis] * phi_v[np.newaxis, :]
    del ii
    step = 0
    lr = 0.1
    plot_initial = True

# set bounary conditions
U[0, :] = U[1, :] = a
V[0, :] = V[1, :] = 0
U[-1, :] = U[-2, :] = phi_u
V[-1, :] = V[-2, :] = phi_v
U[:, 0] = U[:, 1] = U[:, 2]
U[:, -1] = U[:, -2] = U[:, -3]
V[:, 0] = V[:, 1] = 0
V[:, -1] = V[:, -2] = V[:, -3]


def E(UU, VV, s):
    return (s ** 2) * K(UU, VV) + int_Fdz(UU, VV)


# int_e^z [F(U,V)-F(phi)] dz
def int_Fdz(UU, VV):
    # (1/2) * (u_n+1 - n_n-1)^2/(2k)^2 dydz = (...)^2 * (h/k/8)
    Uys = (UU[1:-1, 2:] - UU[1:-1, :-2]) ** 2 * (h / k / 8)
    Vys = (VV[1:-1, 2:] - VV[1:-1, :-2]) ** 2 * (h / k / 8)
    phi_ys = ((phi_u[2:] - phi_u[:-2]) ** 2 + (phi_v[2:] - phi_v[:-2]) ** 2) * (
        h / k / 8
    )

    sum_Fdz = (Uys + Vys - phi_ys[np.newaxis, :]) + (
        W(UU[1:-1, 1:-1], VV[1:-1, 1:-1]) - W(phi_u[1:-1], phi_v[1:-1])[np.newaxis, :]
    ) * h * k
    sum_Fdz *= np.exp(x_mesh[1:-1])[:, np.newaxis]
    return np.sum(sum_Fdz)


# constrain part
def K(UU, VV):
    """
    Kc[u] = \int_D e^z [(u_z^2 +v_z^2)/ 2] dzdy
    """
    Uzs = (UU[2:, 1:-1] - UU[:-2, 1:-1]) ** 2 * (k / h / 8)
    Vzs = (VV[2:, 1:-1] - VV[:-2, 1:-1]) ** 2 * (k / h / 8)

    sE_val = np.sum((Uzs + Vzs) * np.exp(x_mesh[1:-1])[:, np.newaxis])
    return sE_val


# iteration scheme
# Kc(u) = CONSTRAIN <=> minimize |Kc(u) - CONSTRAIN| <br>
# minimize Ec(u) <=> minimuze \int_D e^z( (u_y^2 + v_y^2 - phi_y^2)/2 + W(u,v) - W(phi) ) dzdy

pre_m_U = pre_m_V = pre_n = 0
EPSILON = 1e-10
total_delta = 1
residual = -1
# previous = E(U, V, s)
# now_minimum = previous

y_sec_energy = int_Fdz(U, V)
constrain_part = K(U, V)
s = np.sqrt(-y_sec_energy / constrain_part) if y_sec_energy < 0 else 0

previous = y_sec_energy
now_minimum = previous

energy_decreaing_stop = 0

start_time = time.time()
initial_step = step

if plot_initial:
    new_x = x_mesh[1:-1:10]
    N = int(len(new_x))
    x_slice = slice(0, N)
    t_section = 500
    new_u = U[1:-1:10, t_section]
    new_v = V[1:-1:10, t_section]
    new_u = new_u[x_slice]
    new_v = new_v[x_slice]

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(211)
    ax1.plot(new_x, new_u, label="u")
    ax1.legend()
    ax2 = fig.add_subplot(212)
    ax2.plot(new_x, new_v, label="v")
    ax2.legend()
    ax1.set_title(f"the u,v-plots at step={step}")
    plt.savefig(os.path.join(SAVE_DIR, "UV_y_sec_plots", f"plot_{step:08d}.png"))
    del new_x, new_u, new_v
    plt.close(fig)

while step < 500000:
    step += 1

    if energy_decreaing_stop > 30:
        lr *= 0.95
        energy_decreaing_stop = 0
    if step % 100000 == 0:
        lr *= 1.1

    # penalty = min(10 * lr, 0.5)

    mu = 0.1 * (1.5 + 1.5 * np.exp(-step / 1000))
    nu = 0.1 * (1.5 + 1.5 * np.exp(-step / 1000))

    Uz = U[2:, 1:-1] - U[:-2, 1:-1]
    Vz = V[2:, 1:-1] - V[:-2, 1:-1]

    dU_z = (Uz[:-2, 1:-1] - Uz[2:, 1:-1]) * (k / h / 2)
    dV_z = (Vz[:-2, 1:-1] - Vz[2:, 1:-1]) * (k / h / 2)
    constrain_diff = (
        np.sum(np.exp(x_mesh[1:-1])[:, np.newaxis] * (Uz ** 2 + Vz ** 2) * (k / h / 8))
        - CONSTRAINT
    )
    constrain_propagate_sign = 1 if constrain_diff > 0 else -1
    constrain_diff = np.sqrt(np.abs(constrain_diff))
    constrain_diff = max(1, constrain_diff)

    Uy = U[1:-1, 2:] - U[1:-1, :-2]
    Vy = V[1:-1, 2:] - V[1:-1, :-2]
    p_intF_pU = (Uy[1:-1, :-2] - Uy[1:-1, 2:]) * (h / k / 2) + DuW(
        U[2:-2, 2:-2], V[2:-2, 2:-2]
    ) * (h * k)
    p_intF_pV = (Vy[1:-1, :-2] - Vy[1:-1, 2:]) * (h / k / 2) + DvW(
        U[2:-2, 2:-2], V[2:-2, 2:-2]
    ) * (h * k)

    dU = p_intF_pU / constrain_diff + constrain_propagate_sign * dU_z
    dV = p_intF_pV / constrain_diff + constrain_propagate_sign * dV_z
    dU *= np.exp(x_mesh[2:-2])[:, np.newaxis]
    dV *= np.exp(x_mesh[2:-2])[:, np.newaxis]
    m_U = mu * pre_m_U + (1 - mu) * dU
    m_V = mu * pre_m_V + (1 - mu) * dV

    # this_n = nu * pre_n + (1 - nu) * (np.sum(pncEpU ** 2) + np.sum(pncEpV ** 2))
    this_n = nu * pre_n + (1 - nu) * (np.sum(dU ** 2) + np.sum(dV ** 2))

    if step < 20:
        m_U /= 1 - mu ** step
        m_V /= 1 - mu ** step
        this_n /= 1 - nu ** step

    rand_coefs = 0.5 + np.random.random(size=m_U.shape) * 0.5
    rand_coefs[rand_coefs < 0] = 0

    U_delta = rand_coefs * lr * m_U / (np.sqrt(this_n) + EPSILON)
    V_delta = rand_coefs * lr * m_V / (np.sqrt(this_n) + EPSILON)

    total_delta = np.sum(np.abs(U_delta) + np.abs(V_delta))

    U[2:-2, 2:-2] -= U_delta
    V[2:-2, 2:-2] -= V_delta

    V[V < 0] = 0
    V[V > 1] = 1

    U[0, :] = U[1, :] = a
    V[0, :] = V[1, :] = 0
    U[-1, :] = U[-2, :] = phi_u
    V[-1, :] = V[-2, :] = phi_v
    U[:, 0] = U[:, 1] = U[:, 2]
    U[:, -1] = U[:, -2] = U[:, -3]
    V[:, 0] = V[:, 1] = 0
    V[:, -1] = V[:, -2] = V[:, -3]

    pre_m_U = m_U
    pre_m_V = m_V

    y_sec_energy = int_Fdz(U, V)
    constrain_part = K(U, V)
    s = np.sqrt(-y_sec_energy / constrain_part) if y_sec_energy < 0 else 0

    if y_sec_energy >= now_minimum:
        energy_decreaing_stop += 1
    else:
        energy_decreaing_stop = 0
        now_minimum = y_sec_energy
    residual = y_sec_energy - previous
    previous = y_sec_energy

    now_time = time.time()
    time_duration = now_time - start_time
    if step:
        print(
            f"{step:08d} >",
            f"lr: {lr:.8f};",
            f"Kc[u]: {constrain_part:020.10f};",
            f"y-sec_Energy: {y_sec_energy:020.10f};",
            f"delta: {total_delta:.8f};",
            f"speed: {s:07.4f};",
            f"sec/step: {time_duration/(step-initial_step):05.2f}",
            end="\r",
            flush=True,
        )

    if step % 10 == 0:

        with open(f"{SAVE_DIR}/t_mesh.npy", "wb") as fp:
            np.save(fp, t_mesh)
        with open(f"{SAVE_DIR}/x_mesh.npy", "wb") as fp:
            np.save(fp, x_mesh)
        with open(f"{SAVE_DIR}/U_mesh.npy", "wb") as fp:
            np.save(fp, U)
        with open(f"{SAVE_DIR}/V_mesh.npy", "wb") as fp:
            np.save(fp, V)
        with open(f"{SAVE_DIR}/speed.npy", "wb") as fp:
            np.save(fp, s)

        step_lines = [step, lr, s]
        with open(f"{SAVE_DIR}/steps.json", "w") as f:
            json.dump(step_lines, f, ensure_ascii=False, indent=4)

    if step % 10 == 0:
        now_time = time.time()

        time_duration = now_time - start_time
        minutes = time_duration // 60
        seconds = time_duration - 60 * minutes

        hours, minutes = minutes // 60, minutes % 60
        days, hours = hours // 24, hours % 24
        days, hours, minutes = int(days), int(hours), int(minutes)

        message = (
            f"{step:08d} >"
            + f"lr: {lr:.8f};"
            + f"Kc[u]: {constrain_part:020.10f};"
            + f"y-sec_Energy: {y_sec_energy:020.10f};"
            + f"delta: {total_delta:.8f};"
            + f"speed: {s:07.4f};"
            + f"sec/step: {time_duration/(step-initial_step):05.2f}"
        )

        message_length = len(message)
        print("[" + message + " " + "-" * (142 - message_length) + "]", end="\n")
        log_dict = dict(
            step=step,
            lr=lr,
            y_sec_energy=y_sec_energy,
            constraint=constrain_part,
            speed=s,
        )
        with open(f"{SAVE_DIR}/solve_equation.log", "a") as fp:
            json.dump(log_dict, fp, ensure_ascii=False)
            fp.write("\n")

    if step % 100 == 0:
        SAMPLE_SIZE = 10
        sample_u = U[1:-1:SAMPLE_SIZE, 1:-1:SAMPLE_SIZE]
        sample_v = V[1:-1:SAMPLE_SIZE, 1:-1:SAMPLE_SIZE]
        new_meshsize = h * SAMPLE_SIZE
        uxx_uyy = (
            (sample_u[2:, :] + sample_u[:-2, :] - 2 * sample_u[1:-1, :])
            + (sample_u[:, 2:] + sample_u[:, :-2] - 2 * sample_u[:, 1:-1])
        ) / (new_meshsize ** 2)
        vxx_vyy = (
            (sample_v[2:, :] + sample_v[:-2, :] - 2 * sample_v[1:-1, :])
            + (sample_v[:, 2:] + sample_v[:, :-2] - 2 * sample_v[:, 1:-1])
        ) / (new_meshsize ** 2)
        eq_u = -(s ** 2) * uxx_uyy + DuW(sample_u[2:-2], sample_v[2:-2])
        eq_v = -(s ** 2) * vxx_vyy + DvW(sample_u[2:-2], sample_v[2:-2])

        message = f"eq. error > eq_u: {eq_u.max():.10f} {eq_u.min():.10f}; eq_v: {eq_v.max():.10f} {eq_v.min():.10f}"
        message_length = len(message)
        print("{" + message + " " + "-" * (142 - message_length) + "}", end="\n")

    if step % 100 == 0:
        new_x = x_mesh[1:-1:10]
        N = int(len(new_x))
        x_slice = slice(0, N)
        t_section = 500
        new_u = U[1:-1:10, t_section]
        new_v = V[1:-1:10, t_section]
        new_u = new_u[x_slice]
        new_v = new_v[x_slice]

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(211)
        ax1.plot(new_x, new_u, label="u")
        ax1.legend()
        ax2 = fig.add_subplot(212)
        ax2.plot(new_x, new_v, label="v")
        ax2.legend()
        ax1.set_title(f"the u,v-plots at step={step}")
        plt.savefig(os.path.join(SAVE_DIR, "UV_y_sec_plots", f"plot_{step:08d}.png"))
        del new_x, new_u, new_v
        plt.close(fig)

print("")
print(f"Stop at step = {step} and delta = {total_delta}.")

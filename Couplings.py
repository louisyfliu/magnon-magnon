"""
Created: July 22

@author: yuefei

:keyword: all sensitive parameters as reduced ratio and external field ( \Omega, w_0)
:parameters:   g_ac/g_ab; g_bc/g_ab;(\Omega, w_0/w)
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

"""
The external field 
"""

mu = 1e12  # ??? the magnetic moment length Hz/Tesla  1e11 有好结果 g_ap = 1e9  g_bp = 1e9 g_ab = 1e12

B_0 = 0.1932
# Omega_m = 6e12  # np.sqrt(2 * N * s) * mu * B_0  # 很重要！！！

"""
The dissipation and environment
"""
K_m = 1e9  # The dissipation rate of the magnon modes - MHz

gamma_p = 100  # The mechanical damping rate - Hz

# T at the low temperature; K

hbarKb = 0.673e-11

"""
the partial transposition matrices & the symplectic matrix 
"""
p = np.diag([1, -1, 1, 1])

Omega_2 = np.array([[0, 1, 0, 0],
                    [-1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, -1, 0]])

""" 
Free parameters
"""

w = 1e12  # The frequency of the magnon mode(s) - THz

# w_ratio = 1.5

w0_ratio = 0.07

# w_p = w_ratio * w  # The frequency of the mechanical mode - THz

w_0 = w0_ratio * w  # The drive magnetic field frequency (microwave) ; THz

g_ab = 1e12  # magnon-magnon coupling rate; THz


# g_mp = g_ap = g_bp = g_ratio * g_ab the magnon-phonon coupling rate; THz 0.001


def get_q_s_given_param(g_ratio_ac, g_ratio_bc, w_ratio, i):
    """
    Compute the steady state solution q_s given parameters
    """

    g_ap = g_ratio_ac * g_ab

    g_bp = g_ratio_bc * g_ab

    w_p = w_ratio * w

    w_a = w - w_0  # Detuning frequency of A mode

    w_b = w + w_0  # Detuning frequency of B mode

    Omega_m = mu * B_0

    a5 = w_p * g_ap ** 2 * g_bp ** 2
    a4 = 2 * w_b * w_p * g_ap ** 2 * g_bp + 2 * w_a * w_p * g_ap * g_bp ** 2 + Omega_m ** 2 * g_ap ** 3 * g_bp ** 2
    a3 = w_b ** 2 * w_p * g_ap ** 2 + w_a ** 2 * w_p * g_bp ** 2 + 4 * w_a * w_b * w_p * g_ab * g_bp - 2 * w_p * g_ap * g_bp * g_ab ** 2 \
         + 2 * Omega_m ** 2 * w_p * (w_b + g_ab) * g_ap ** 3 * g_bp + 2 * Omega_m ** 2 * w_p * (
                 w_a + g_ab) * g_ap ** 2 * g_bp ** 2
    a2 = 2 * w_a * w_b ** 2 * w_p * g_ap + 2 * w_a ** 2 * w_b * w_p * g_bp - 2 * w_a * w_p * g_bp * g_ab ** 2 - 2 * w_b * w_p * g_ap * g_ab ** 2 \
         + Omega_m ** 2 * (w_a + g_ab) ** 2 * g_ap * g_bp ** 2 + Omega_m ** 2 * (w_b + g_ab) ** 2 * g_ap ** 3 \
         + 4 * Omega_m ** 2 * (w_a + g_ab) * (w_b + g_ab) * g_ap ** 2 * g_bp
    a1 = w_p * g_ab ** 2 + w_a ** 2 * w_b ** 2 * w_p - 2 * w_a * w_b * w_p * g_ab ** 2 + 2 * Omega_m ** 2 * (
            w_a + g_ab) * (w_b + g_ab) ** 2 * g_ap ** 2 \
         + 2 * Omega_m ** 2 * (w_a + g_ab) ** 2 * (w_b + g_ab) * g_ap * g_bp
    a0 = Omega_m ** 2 * (w_a + g_ab) ** 2 * g_bp + Omega_m ** 2 * (w_b + g_ab) ** 2 * g_ap

    p = np.poly1d([a5, a4, a3, a2, a1, a0])
    # print(p)
    rootsp = p.r

    # print("\nRoots of the Polynomial is :", rootsp)

    q = rootsp[i - 1]

    # print("\nHere we pick :", q)
    # print("\nThe format of the data is :", type(q))

    return q


def get_a_s_given_qi(g_ratio_ac, g_ratio_bc, q):
    """
    Compute the steady state solution a_s & b_s given q_s
    """

    g_ap = g_ratio_ac * g_ab

    g_bp = g_ratio_bc * g_ab

    w_a = w - w_0  # Detuning frequency of A mode

    w_b = w + w_0  # Detuning frequency of B mode

    Omega_m = mu * B_0

    Delta_ap = w_a + (g_ap * q)
    Delta_bp = w_b + (g_bp * q)
    a_s = -1j * Omega_m * (Delta_bp + g_ab) / (Delta_ap * Delta_bp - (g_ab ** 2))

    # print("\nSteady state solution of a is :", a_s)
    # print(type(a_s))
    return a_s


def get_b_s_given_qi(g_ratio_ac, g_ratio_bc, q):
    """
    Compute the steady state solution a_s & b_s given q_s
    """

    g_ap = g_ratio_ac * g_ab

    g_bp = g_ratio_bc * g_ab

    w_a = w - w_0  # Detuning frequency of A mode

    w_b = w + w_0  # Detuning frequency of B mode

    Omega_m = mu * B_0

    Delta_ap = w_a + (g_ap * q)
    Delta_bp = w_b + (g_bp * q)
    b_s = -1j * Omega_m * (Delta_ap + g_ab) / (Delta_ap * Delta_bp - (g_ab ** 2))

    # print("\nSteady state solution of b is :", b_s)
    # print(type(b_s))
    return b_s


def get_A_given_steady_states_solutions(a, b, q, g_ratio_ac, g_ratio_bc, w_ratio):
    """
    Compute the drift matrix A given steady state solution
    """

    g_ap = g_ratio_ac * g_ab

    g_bp = g_ratio_bc * g_ab

    w_p = w_ratio * w

    w_a = w - w_0  # Detuning frequency of A mode

    w_b = w + w_0  # Detuning frequency of B mode

    """
    The notations 
    """
    Delta_ap = w_a + g_ap * q

    Delta_bp = w_b + g_bp * q

    M0 = g_ab  # g_ab + g_abp * q[i]

    M1 = np.sqrt(2) * (g_ap * a)  # + complex(g_abp * b))

    M2 = np.sqrt(2) * (g_bp * b)  # + complex(g_abp * a)

    """
    The drift matrix  
    """
    A = np.array([[-K_m, Delta_ap, -M0.imag, -M0.real, M1.imag, 0],
                  [-Delta_ap, -K_m, -M0.real, M0.imag, -M1.real, 0],
                  [-M0.imag, -M0.real, -K_m, Delta_bp, M2.imag, 0],
                  [-M0.real, M0.imag, -Delta_bp, -K_m, -M2.real, 0],
                  [0, 0, 0, 0, 0, w_p],
                  [-M1.real, -M1.imag, -M2.real, -M2.imag, -w_p, gamma_p]])
    # print(A)
    return A


def get_V_given_A(T, A, w_ratio):
    """
    Solve the value V using Lyapunov Equation
    """
    w_p = w_ratio * w

    w_a = w - w_0  # Detuning frequency of A mode

    w_b = w + w_0  # Detuning frequency of B mode

    # Compute N_m & N_p
    N_a = 1 / (np.exp(w_a * hbarKb / T) - 1)  # The equilibrium mean thermal magnon number
    N_b = 1 / (np.exp(w_b * hbarKb / T) - 1)  # The equilibrium mean thermal magnon number
    N_p = 1 / (np.exp(w_p * hbarKb / T) - 1)  # The equilibrium mean thermal phonon number

    # Compute the diagonal elements D1 & D2 & D3
    D1 = K_m * (2 * N_a + 1)
    D2 = K_m * (2 * N_b + 1)
    D3 = gamma_p * (2 * N_p + 1)

    # Form the matrix D
    D = np.array([[D1, 0, 0, 0, 0, 0],
                  [0, D1, 0, 0, 0, 0],
                  [0, 0, D2, 0, 0, 0],
                  [0, 0, 0, D2, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, D3]])

    # Solve the equation to get V
    V = linalg.solve_continuous_lyapunov(A, D)
    return V


def get_LN_given_V_0(V):
    """
    'potential' logarithmic negativity of bipartite
    """
    v = np.array(V)

    def submatrix(startRow, startCol, size):
        return v[startRow:startRow + size, startCol:startCol + size]

    V4 = submatrix(0, 0, 4)
    V_0 = p.dot(V4).dot(p)
    V_1 = 1j * Omega_2.dot(V_0)
    Evals, Eves = linalg.eig(V_1)
    # print(Evals)
    # print(Eves)
    AbsEvals = np.abs([np.abs(i) for i in Evals])
    # print(AbsEvals)
    V_2 = np.min([np.min(arr) for arr in AbsEvals])
    V_3 = round(V_2, 7)
    # print(V_3)
    L_N = -np.log(2 * V_3)
    return L_N


def get_LN_given_V_1(V):
    """
    'potential' logarithmic negativity of bipartite
    """
    v = np.array(V)

    def submatrix(startRow, startCol, size):
        return v[startRow:startRow + size, startCol:startCol + size]

    V4 = submatrix(2, 2, 4)
    V_0 = p.dot(V4).dot(p)
    V_1 = 1j * Omega_2.dot(V_0)
    Evals, Eves = linalg.eig(V_1)
    # print(Evals)
    # print(Eves)
    AbsEvals = np.abs([np.abs(i) for i in Evals])
    # print(AbsEvals)
    V_2 = np.min([np.min(arr) for arr in AbsEvals])
    V_3 = round(V_2, 7)
    # print(V_3)
    L_N = -np.log(2 * V_3)
    return L_N


def get_LN_given_V_2(V):
    """
    'potential' logarithmic negativity of bipartite
    """
    # print(V)
    Y1 = V[0:2, 0:2]
    Y2 = V[4:6, 0:2]
    Y3 = V[0:2, 4:6]
    Y4 = V[4:6, 4:6]
    X1 = np.concatenate((Y1, Y2), axis=0)
    X2 = np.concatenate((Y3, Y4), axis=0)
    V4 = np.concatenate((X1, X2), axis=1)
    # print(V4)
    V_0 = p.dot(V4).dot(p)
    V_1 = 1j * Omega_2.dot(V_0)
    Evals, Eves = linalg.eig(V_1)
    # print(Evals)
    # print(Eves)
    AbsEvals = np.abs([np.abs(i) for i in Evals])
    # print(AbsEvals)
    V_2 = np.min([np.min(arr) for arr in AbsEvals])
    V_3 = round(V_2, 7)
    # print(V_3)
    L_N = -np.log(2 * V_3)
    return L_N


def solve_E_N_0(g_ratio_ac, g_ratio_bc, i, T, w_ratio):
    q = get_q_s_given_param(g_ratio_ac, g_ratio_bc, w_ratio, i)
    a = get_a_s_given_qi(g_ratio_ac, g_ratio_bc, q)
    b = get_b_s_given_qi(g_ratio_ac, g_ratio_bc, q)
    A = get_A_given_steady_states_solutions(a, b, q, g_ratio_ac, g_ratio_bc, w_ratio)
    V = get_V_given_A(T, A, w_ratio)
    L_N = get_LN_given_V_0(V)
    E_N = max(0, L_N)
    return E_N


def solve_E_N_1(g_ratio_ac, g_ratio_bc, i, T, w_ratio):
    q = get_q_s_given_param(g_ratio_ac, g_ratio_bc, w_ratio, i)
    a = get_a_s_given_qi(g_ratio_ac, g_ratio_bc, q)
    b = get_b_s_given_qi(g_ratio_ac, g_ratio_bc, q)
    A = get_A_given_steady_states_solutions(a, b, q, g_ratio_ac, g_ratio_bc, w_ratio)
    V = get_V_given_A(T, A, w_ratio)
    L_N = get_LN_given_V_1(V)
    E_N = max(0, L_N)
    return E_N


def solve_E_N_2(g_ratio_ac, g_ratio_bc, i, T, w_ratio):
    q = get_q_s_given_param(g_ratio_ac, g_ratio_bc, w_ratio, i)
    a = get_a_s_given_qi(g_ratio_ac, g_ratio_bc, q)
    b = get_b_s_given_qi(g_ratio_ac, g_ratio_bc, q)
    A = get_A_given_steady_states_solutions(a, b, q, g_ratio_ac, g_ratio_bc, w_ratio)
    V = get_V_given_A(T, A, w_ratio)
    L_N = get_LN_given_V_2(V)
    E_N = max(0, L_N)
    return E_N


if __name__ == '__main__':

    N = 100
    M = 100

    array_g_ratio_ac = np.linspace(5e-4, 2e-3, M)
    array_g_ratio_bc = np.linspace(5e-4, 2e-3, N)

    '''
    create the meshgrid matrix of E_N
    '''


    def get_EN1_matrix():
        EN = np.zeros((N, M))
        T = 100
        for i in range(N):
            for j in range(M):
                EN[i][j] = solve_E_N_0(array_g_ratio_ac[i], array_g_ratio_bc[j], 5, T, 0.75)
                if EN[i][j] == 0:
                    EN[i][j] = solve_E_N_0(array_g_ratio_ac[i], array_g_ratio_bc[j], 5, T, 1)
                    if EN[i][j] == 0:
                        EN[i][j] = solve_E_N_0(array_g_ratio_ac[i], array_g_ratio_bc[j], 5, T, 1.25)
                    else:
                        EN[i][j] = solve_E_N_0(array_g_ratio_ac[i], array_g_ratio_bc[j], 5, T, 1)
                else:
                    EN[i][j] = solve_E_N_0(array_g_ratio_ac[i], array_g_ratio_bc[j], 5, T, 0.75)
        return EN


    plt.rcParams.update({'font.size': 16})  # font size

    fig, ax = plt.subplots()
    surf = ax.contourf(array_g_ratio_ac, array_g_ratio_bc, get_EN1_matrix(), levels=[-1, 0, 10],
                       colors=['white', 'blue'])

    # surf2 = ax.contourf(array_g_ratio_ac, array_g_ratio_bc,  levels=[-1, 0, 4], colors=['white', 'red'])

    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('$g_{ac}/g_{ab}$')
    ax.set_ylabel('$g_{bc}/g_{ab}$')

    # plt.title('$( \omega_0/\omega = 0.1; \ \Omega=0.57 \ THz; \ T = 50K )$')
    plt.show()

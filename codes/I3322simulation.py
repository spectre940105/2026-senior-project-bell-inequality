import numpy as np
from scipy.optimize import differential_evolution

#----------------定義基礎量子態與矩陣與全域變數----------------
def tensor(a,b): return np.kron(a,b)
#Def alpha0 = 跟s.t. 產生最大糾纏態
alpha0 = np.sqrt(2)/2


ket0 = np.array([[1], [0]])
ket1 = np.array([[0], [1]])

ket00 = tensor(ket0, ket0)
ket11 = tensor(ket1, ket1)
I = np.eye(2)

#Pauli Matrix
sigma_x = np.array([[0,1], [1,0]])
sigma_y = np.array([[0,-1j], [1j,0]])
sigma_z = np.array([[1,0], [0,-1]])
#--------------------------------------------------


#-------------糾纏態、密度矩陣、投影算符-------------
def entangled_state(alpha):
    return alpha * ket00 + np.sqrt(1 - alpha ** 2) * ket11

def density_matrix(alpha):
    return entangled_state(alpha) @ entangled_state(alpha).conj().T

def projection(theta , phi):
    #布洛赫球面角度換算座標
    vec = np.array([np.sin(theta) * np.cos(phi) , np.sin(theta) * np.sin(phi) , np.cos(theta)])
    vecdotpauli = vec[0] * sigma_x + vec[1] * sigma_y + vec[2] * sigma_z
    return (I + vecdotpauli) / 2
#--------------------------------------------------


#-----------------邊際機率&聯合機率-----------------
def PA(rho,A): return np.real(np.trace(rho @ tensor(A, I)))
def PB(rho,B): return np.real(np.trace(rho @ tensor(I, B)))
def PAB(rho,A,B): return np.real(np.trace(rho @ tensor(A, B)))
#--------------------------------------------------


#-------------------I3322 不等式-------------------
def I3322(rho, A1, A2, A3, B1, B2, B3):
    Marginal_probability = -2 * PA(rho, A1) - PA(rho, A2) - PB(rho, B1)
    joint_probability = (
          PAB(rho, A1, B1) + PAB(rho, A1, B2) + PAB(rho, A1, B3)
        + PAB(rho, A2, B1) + PAB(rho, A2, B2) - PAB(rho, A2, B3)
        + PAB(rho, A3, B1) - PAB(rho, A3, B2) + 0
    )
    return Marginal_probability + joint_probability
#--------------------------------------------------

#--------------------找最佳角度--------------------
def objective_func(params,rho):
    #12個隨機角度分配給A1~B3
    A1 = projection(params[0],params[1])
    A2 = projection(params[2],params[3])
    A3 = projection(params[4],params[5])
    B1 = projection(params[6],params[7])
    B2 = projection(params[8],params[9])
    B3 = projection(params[10],params[11])

    score = I3322(rho, A1, A2, A3, B1, B2, B3)

    return -score #找最小值 = 找最大違背值

if __name__ == '__main__':
    rho_text = density_matrix(alpha0)

    #設定搜尋範圍 theta 0~pi phi 0~2pi
    bounds = [(0.0,np.pi),(0.0,2*np.pi)]*6

    result = differential_evolution(
        func=objective_func,
        bounds=bounds,
        args=(rho_text,),
        seed=32,
        popsize=30,
        maxiter=500,
        tol=1e-8,
        polish=True,
        workers=-1,
        updating='deferred'
    )

    best_angles = result.x
    max_violation = -result.fun

    print(f'最大違背值 : {max_violation:.6f}',flush=True)
    labels = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']
    for i in range(6):
        theta = best_angles[2*i]
        phi = best_angles[2*i+1]
        print(f"{labels[i]}: θ = {theta:.4f} 弧度, φ = {phi:.4f} 弧度",flush=True)
#--------------------------------------------------


#--------------------數值模擬--------------------
def joint_outcome_probabilities(rho, A, B):
    """計算 (A, B) 的四種聯合結果 (0,0)、(0,1)、(1,0)、(1,1)。"""
    not_A = I - A
    not_B = I - B
    probabilities = np.array([
        PAB(rho, not_A, not_B),
        PAB(rho, not_A, B),
        PAB(rho, A, not_B),
        PAB(rho, A, B),
    ])

    probabilities = np.clip(probabilities, 0.0, 1.0)
    return probabilities / probabilities.sum()


def simulate_I3322(rho, angles, shots = 12500, rng=None):

    A1 = projection(angles[0],angles[1])
    A2 = projection(angles[2],angles[3])
    A3 = projection(angles[4],angles[5])
    B1 = projection(angles[6],angles[7])
    B2 = projection(angles[8],angles[9])
    B3 = projection(angles[10],angles[11])

    if rng is None:
        rng = np.random.default_rng()

    counts = {
        'A1B1': rng.multinomial(shots, joint_outcome_probabilities(rho, A1, B1)),
        'A1B2': rng.multinomial(shots, joint_outcome_probabilities(rho, A1, B2)),
        'A1B3': rng.multinomial(shots, joint_outcome_probabilities(rho, A1, B3)),
        'A2B1': rng.multinomial(shots, joint_outcome_probabilities(rho, A2, B1)),
        'A2B2': rng.multinomial(shots, joint_outcome_probabilities(rho, A2, B2)),
        'A2B3': rng.multinomial(shots, joint_outcome_probabilities(rho, A2, B3)),
        'A3B1': rng.multinomial(shots, joint_outcome_probabilities(rho, A3, B1)),
        'A3B2': rng.multinomial(shots, joint_outcome_probabilities(rho, A3, B2)),
    }

    sim_pA1B1 = counts['A1B1'][3] / shots
    sim_pA1B2 = counts['A1B2'][3] / shots
    sim_pA1B3 = counts['A1B3'][3] / shots
    sim_pA2B1 = counts['A2B1'][3] / shots
    sim_pA2B2 = counts['A2B2'][3] / shots
    sim_pA2B3 = counts['A2B3'][3] / shots
    sim_pA3B1 = counts['A3B1'][3] / shots
    sim_pA3B2 = counts['A3B2'][3] / shots

    sim_pA1 = (counts['A1B1'][2] + counts['A1B1'][3] +
               counts['A1B2'][2] + counts['A1B2'][3] +
               counts['A1B3'][2] + counts['A1B3'][3]) / (3 * shots)
    sim_pA2 = (counts['A2B1'][2] + counts['A2B1'][3] +
               counts['A2B2'][2] + counts['A2B2'][3] +
               counts['A2B3'][2] + counts['A2B3'][3]) / (3 * shots)
    sim_pB1 = (counts['A1B1'][1] + counts['A1B1'][3] +
               counts['A2B1'][1] + counts['A2B1'][3] +
               counts['A3B1'][1] + counts['A3B1'][3]) / (3 * shots)

    sim_marginal = - 2 * sim_pA1 - sim_pA2 - sim_pB1
    sim_joint = (
        sim_pA1B1 + sim_pA1B2 + sim_pA1B3
      + sim_pA2B1 + sim_pA2B2 - sim_pA2B3
      + sim_pA3B1 - sim_pA3B2 + 0
    )
    return sim_marginal + sim_joint
#--------------------------------------------------


#--------------------混合糾纏態--------------------
def K1(p):
    return np.array([
        [1, 0],
        [0, np.sqrt(1-p)]
    ])

def K2(p,eta):
    return np.array([
        [0, eta * np.sqrt(p)],
        [0, np.sqrt(1 - eta**2) * np.sqrt(p)]
    ])

def epsilon(rho,p,eta):
    K1_double = tensor(I, K1(p))
    K2_double = tensor(I, K2(p,eta))

    rho_prime = np.zeros_like(rho,dtype=complex)
    rho_prime = rho_prime + (K1_double @ rho @ K1_double.conj().T)
    rho_prime = rho_prime + (K2_double @ rho @ K2_double.conj().T)

    return rho_prime
#--------------------------------------------------

if __name__ == '__main__':
    A1 = projection(best_angles[0],best_angles[1])
    A2 = projection(best_angles[2],best_angles[3])
    A3 = projection(best_angles[4],best_angles[5])
    B1 = projection(best_angles[6],best_angles[7])
    B2 = projection(best_angles[8],best_angles[9])
    B3 = projection(best_angles[10],best_angles[11])

    #圖 1(a) 資料收集
    rounds_1a = 100
    data_1a = {}
    alpha_list = [alpha0 , 0.1 , 0.3 , 0.5 , 0.9]

    for a in alpha_list:
        rho = density_matrix(a)
        history = []

        for _ in range(rounds_1a):
            history.append(simulate_I3322(rho,best_angles))
        data_1a[a] = history

    #圖 1(b) 資料收集
    alphas_1b = np.linspace(0,1,40)
    theory_1b = []
    sim_1b = []

    for a in alphas_1b:
        rho = density_matrix(a)

        theory_1b.append(I3322(rho, A1, A2, A3, B1, B2, B3))

        sim_1b.append(simulate_I3322(rho,best_angles))


    #對應文獻圖.3(a)(b) - 固定角度實驗版
    grid_size = 50
    p_value = np.linspace(0, 1, grid_size)
    eta_value = np.linspace(0, 1, grid_size)

    I3322_grid = np.zeros((grid_size, grid_size))
    rho_initial = density_matrix(alpha0)


    # 2. 雙重迴圈：直接代入公式
    for i in range(grid_size):
        for j in range(grid_size):
            p = p_value[i]
            eta = eta_value[j]
            rho_noise = epsilon(rho_initial, p, eta)
            I3322_grid[i, j] = I3322(rho_noise, A1, A2, A3, B1, B2, B3)

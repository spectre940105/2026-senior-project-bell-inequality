import numpy as np
from scipy.optimize import differential_evolution

#----------------定義基礎量子態與矩陣與全域變數----------------
def tensor(a,b): return np.kron(a,b)
#Def alpha0 s.t. 產生最大糾纏態
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


#-------------------I2222 不等式-------------------
def I2222(rho, A1, A2, B1, B2):
    marginal_probability = -PA(rho,A1) - PB(rho,B1)
    joint_probability = (
        PAB(rho,A1,B1) + PAB(rho,A1,B2)
      + PAB(rho,A2,B1) - PAB(rho,A2,B2)
                        )
    return marginal_probability + joint_probability
#--------------------------------------------------


#--------------------找最佳角度--------------------
def objective_func(params,rho):
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


#--------------------I3322數值模擬--------------------
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


def simulate_I3322_I2222(
    rho, angles, shots=100000//8, rng=None, return_details=False
):
    """Estimate I3322 and I2222 from one shared Monte Carlo data set."""
    measurements_A = {
        'A1': projection(angles[0], angles[1]),
        'A2': projection(angles[2], angles[3]),
        'A3': projection(angles[4], angles[5]),
    }
    measurements_B = {
        'B1': projection(angles[6], angles[7]),
        'B2': projection(angles[8], angles[9]),
        'B3': projection(angles[10], angles[11]),
    }

    if rng is None:
        rng = np.random.default_rng()

    counts = {}
    for a_name, A in measurements_A.items():
        for b_name, B in measurements_B.items():
            setting = a_name + b_name
            probabilities = joint_outcome_probabilities(rho, A, B)
            counts[setting] = rng.multinomial(shots, probabilities)

    def joint_probability(setting):
        return counts[setting][3] / shots

    def alice_marginal(a_name, b_names):
        total = sum(counts[a_name + b][2] + counts[a_name + b][3] for b in b_names)
        return total / (len(b_names) * shots)

    def bob_marginal(b_name, a_names):
        total = sum(counts[a + b_name][1] + counts[a + b_name][3] for a in a_names)
        return total / (len(a_names) * shots)

    pA1_I3322 = alice_marginal('A1', ('B1', 'B2', 'B3'))
    pA2_I3322 = alice_marginal('A2', ('B1', 'B2', 'B3'))
    pB1_I3322 = bob_marginal('B1', ('A1', 'A2', 'A3'))
    I3322_value = (
        -2 * pA1_I3322 - pA2_I3322 - pB1_I3322
        + joint_probability('A1B1') + joint_probability('A1B2') + joint_probability('A1B3')
        + joint_probability('A2B1') + joint_probability('A2B2') - joint_probability('A2B3')
        + joint_probability('A3B1') - joint_probability('A3B2')
    )

    pA1_I2222 = alice_marginal('A1', ('B1', 'B2'))
    pB1_I2222 = bob_marginal('B1', ('A1', 'A2'))
    I2222_value = (
        -pA1_I2222 - pB1_I2222
        + joint_probability('A1B1') + joint_probability('A1B2')
        + joint_probability('A2B1') - joint_probability('A2B2')
    )

    if return_details:
        details = {
            'shots_per_setting': shots,
            'I3322': {
                'P(A1=1)': pA1_I3322,
                'P(A2=1)': pA2_I3322,
                'P(B1=1)': pB1_I3322,
                'P(A1=1,B1=1)': joint_probability('A1B1'),
                'P(A1=1,B2=1)': joint_probability('A1B2'),
                'P(A1=1,B3=1)': joint_probability('A1B3'),
                'P(A2=1,B1=1)': joint_probability('A2B1'),
                'P(A2=1,B2=1)': joint_probability('A2B2'),
                'P(A2=1,B3=1)': joint_probability('A2B3'),
                'P(A3=1,B1=1)': joint_probability('A3B1'),
                'P(A3=1,B2=1)': joint_probability('A3B2'),
            },
            'I2222': {
                'P(A1=1)': pA1_I2222,
                'P(B1=1)': pB1_I2222,
                'P(A1=1,B1=1)': joint_probability('A1B1'),
                'P(A1=1,B2=1)': joint_probability('A1B2'),
                'P(A2=1,B1=1)': joint_probability('A2B1'),
                'P(A2=1,B2=1)': joint_probability('A2B2'),
            },
        }
        return I3322_value, I2222_value, details

    return I3322_value, I2222_value


def print_bell_probability_report(I3322_value, I2222_value, details, seed):
    """Print probabilities, coefficients, and contributions for both inequalities."""
    coefficient_tables = {
        'I3322': (
            ('P(A1=1)', -2),
            ('P(A2=1)', -1),
            ('P(B1=1)', -1),
            ('P(A1=1,B1=1)', 1),
            ('P(A1=1,B2=1)', 1),
            ('P(A1=1,B3=1)', 1),
            ('P(A2=1,B1=1)', 1),
            ('P(A2=1,B2=1)', 1),
            ('P(A2=1,B3=1)', -1),
            ('P(A3=1,B1=1)', 1),
            ('P(A3=1,B2=1)', -1),
        ),
        'I2222': (
            ('P(A1=1)', -1),
            ('P(B1=1)', -1),
            ('P(A1=1,B1=1)', 1),
            ('P(A1=1,B2=1)', 1),
            ('P(A2=1,B1=1)', 1),
            ('P(A2=1,B2=1)', -1),
        ),
    }
    bell_values = {'I3322': I3322_value, 'I2222': I2222_value}

    print("\n" + "=" * 72)
    print(f"共同蒙地卡羅抽樣機率報告：seed={seed}")
    print(f"每組測量設定 shots={details['shots_per_setting']}")
    print("量子態：最大糾纏態；測量角度：I3322 最佳角度")

    for bell_name in ('I3322', 'I2222'):
        print("\n" + f"[{bell_name}]")
        print(f"{'機率項':<22}{'機率值':>14}{'係數':>8}{'貢獻':>14}")
        print("-" * 58)
        for term_name, coefficient in coefficient_tables[bell_name]:
            probability = details[bell_name][term_name]
            contribution = coefficient * probability
            print(
                f"{term_name:<22}{probability:>14.8f}"
                f"{coefficient:>8d}{contribution:>14.8f}"
            )
        print("-" * 58)
        print(f"{bell_name + ' 總值':<44}{bell_values[bell_name]:>14.8f}")
    print("=" * 72 + "\n")
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
    monte_carlo_seed = 32
    monte_carlo_rng = np.random.default_rng(monte_carlo_seed)

    report_I3322, report_I2222, probability_details = simulate_I3322_I2222(
        density_matrix(alpha0),
        best_angles,
        rng=np.random.default_rng(monte_carlo_seed),
        return_details=True,
    )
    print_bell_probability_report(
        report_I3322,
        report_I2222,
        probability_details,
        monte_carlo_seed,
    )

    #圖 1(a) 資料收集
    rounds_1a = 100
    I3322data_1a = {}
    I2222data_1a = {}
    alpha_list = [alpha0 , 0.1 , 0.3 , 0.5 , 0.9]

    for a in alpha_list:
        rho = density_matrix(a)
        I3322history = []
        I2222history = []

        for _ in range(rounds_1a):
            I3322sample, I2222sample = simulate_I3322_I2222(
                rho, best_angles, rng=monte_carlo_rng
            )
            I3322history.append(I3322sample)
            I2222history.append(I2222sample)
        I3322data_1a[a] = I3322history
        I2222data_1a[a] = I2222history

    #圖 1(b) 資料收集
    alphas_1b = np.linspace(0,1,40)
    I3322theory_1b = []
    I2222theory_1b = []
    I3322sim_1b = []
    I2222sim_1b = []

    for a in alphas_1b:
        rho = density_matrix(a)

        I3322theory_1b.append(I3322(rho, A1, A2, A3, B1, B2, B3))
        I2222theory_1b.append(I2222(rho, A1, A2, B1, B2))

        I3322sample, I2222sample = simulate_I3322_I2222(
            rho, best_angles, rng=monte_carlo_rng
        )
        I3322sim_1b.append(I3322sample)
        I2222sim_1b.append(I2222sample)


    #對應文獻圖.3(a)(b) - 固定角度實驗版
    grid_size = 50
    p_value = np.linspace(0, 1, grid_size)
    eta_value = np.linspace(0, 1, grid_size)

    I3322_grid = np.zeros((grid_size, grid_size))
    I2222_grid = np.zeros((grid_size, grid_size))
    rho_initial = density_matrix(alpha0)

    # 2. 雙重迴圈：直接代入公式
    for i in range(grid_size):
        for j in range(grid_size):
            p = p_value[i]
            eta = eta_value[j]
            rho_noise = epsilon(rho_initial, p, eta)
            I3322_grid[i, j] = I3322(rho_noise, A1, A2, A3, B1, B2, B3)
            I2222_grid[i, j] = I2222(rho_noise, A1, A2, B1, B2)




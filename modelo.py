import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
N_MUESTRAS = 5000
CP_C = 4000.0
CP_H = 4180.0
VARIABLES = ['m_c', 'm_h', 'T_c_in', 'T_h_in', 'U', 'A']


def mltd_contracorriente(q_W, m_c, m_h, T_c_in, T_h_in, U, A,
                         cp_c=CP_C, cp_h=CP_H):
    C_c, C_h = m_c * cp_c, m_h * cp_h
    T_c_out = T_c_in + q_W / C_c
    T_h_out = T_h_in - q_W / C_h
    dT1 = T_h_in - T_c_out
    dT2 = T_h_out - T_c_in
    if dT1 <= 0 or dT2 <= 0:
        return np.nan
    if abs(dT1 - dT2) < 1e-12:
        return 0.5 * (dT1 + dT2)
    return (dT1 - dT2) / np.log(dT1 / dT2)


def f_q(q_W, m_c, m_h, T_c_in, T_h_in, U, A,
        cp_c=CP_C, cp_h=CP_H):
    return q_W - U * A * mltd_contracorriente(
        q_W, m_c, m_h, T_c_in, T_h_in, U, A, cp_c, cp_h
    )


def resolver_biseccion(m_c, m_h, T_c_in, T_h_in, U, A,
                       cp_c=CP_C, cp_h=CP_H,
                       tol_q=1e-7, tol_f=1e-4, max_iter=200,
                       guardar_historial=False):
    if min(m_c, m_h, U, A) <= 0 or T_h_in <= T_c_in:
        raise ValueError('Los caudales, U y A deben ser positivos y T_h_in > T_c_in.')
    C_c, C_h = m_c * cp_c, m_h * cp_h
    q_max = min(C_c, C_h) * (T_h_in - T_c_in)
    a, b = 0.0, q_max * (1 - 1e-10)
    fa = f_q(a, m_c, m_h, T_c_in, T_h_in, U, A, cp_c, cp_h)
    fb = f_q(b, m_c, m_h, T_c_in, T_h_in, U, A, cp_c, cp_h)
    if not np.isfinite(fa) or not np.isfinite(fb) or fa * fb > 0:
        raise ValueError('Intervalo inválido para el método de bisección.')
    historial = []
    for iteracion in range(1, max_iter + 1):
        m = 0.5 * (a + b)
        fm = f_q(m, m_c, m_h, T_c_in, T_h_in, U, A, cp_c, cp_h)
        if guardar_historial:
            historial.append({'Iteración': iteracion, 'Q_a (kW)': a/1000,
                              'Q_b (kW)': b/1000, 'Q_m (kW)': m/1000,
                              'Residuo (W)': fm})
        # Conserva el criterio de parada (OR) de la implementación original.
        if abs(fm) < tol_f or abs(b - a) < tol_q:
            break
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    tc_out, th_out = T_c_in + m/C_c, T_h_in - m/C_h
    if guardar_historial:
        return m, tc_out, th_out, iteracion, pd.DataFrame(historial)
    return m, tc_out, th_out, iteracion


def resolver_epsilon_ntu(m_c, m_h, T_c_in, T_h_in, U, A,
                         cp_c=CP_C, cp_h=CP_H):
    C_c, C_h = m_c * cp_c, m_h * cp_h
    C_min, C_max = min(C_c, C_h), max(C_c, C_h)
    C_r = C_min / C_max
    ntu = U * A / C_min
    if abs(1 - C_r) < 1e-12:
        epsilon = ntu / (1 + ntu)
    else:
        x = np.exp(-ntu * (1 - C_r))
        epsilon = (1 - x) / (1 - C_r * x)
    return epsilon * C_min * (T_h_in - T_c_in)


def generar_dataset():
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({
        'm_c': rng.uniform(0.5, 3.0, N_MUESTRAS),
        'm_h': rng.uniform(0.5, 3.0, N_MUESTRAS),
        'T_c_in': rng.uniform(20, 50, N_MUESTRAS),
        'T_h_in': rng.uniform(70, 140, N_MUESTRAS),
        'U': rng.uniform(300, 1000, N_MUESTRAS),
        'A': rng.uniform(5, 25, N_MUESTRAS),
        't_h': rng.uniform(0.5, 8.0, N_MUESTRAS),
        'eta': rng.uniform(0.75, 0.95, N_MUESTRAS),
    })
    resultados = [resolver_biseccion(f.m_c, f.m_h, f.T_c_in,
                                     f.T_h_in, f.U, f.A)
                  for f in df.itertuples(index=False)]
    result = np.asarray(resultados)
    df['Q_kW'] = result[:, 0]/1000
    df['T_c_out'] = result[:, 1]
    df['T_h_out'] = result[:, 2]
    df['bisection_iters'] = result[:, 3].astype(int)
    df['Q_NTU_kW'] = [resolver_epsilon_ntu(f.m_c, f.m_h, f.T_c_in,
                                          f.T_h_in, f.U, f.A)/1000
                      for f in df.itertuples(index=False)]
    df['E_thermal_kWh'] = df['Q_kW'] * df['t_h']
    df['E_cons_kWh'] = df['E_thermal_kWh'] / df['eta']
    df['error_pct'] = abs(df['Q_kW'] - df['Q_NTU_kW']) / df['Q_NTU_kW'] * 100
    return df


def entrenar_modelo(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df[VARIABLES], df['Q_kW'], test_size=0.20, random_state=SEED
    )
    rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    q_pred = rf.predict(X_test)
    prueba = df.loc[X_test.index]
    e_pred = q_pred * prueba['t_h'].to_numpy() / prueba['eta'].to_numpy()
    e_true = prueba['E_cons_kWh'].to_numpy()
    metricas = {
        'Q_MAE': mean_absolute_error(y_test, q_pred),
        'Q_RMSE': np.sqrt(mean_squared_error(y_test, q_pred)),
        'Q_R2': r2_score(y_test, q_pred),
        'E_MAE': mean_absolute_error(e_true, e_pred),
        'E_RMSE': np.sqrt(mean_squared_error(e_true, e_pred)),
        'E_R2': r2_score(e_true, e_pred),
        'E_MAPE': np.mean(abs((e_true - e_pred)/e_true))*100,
    }
    return rf, metricas, pd.DataFrame({'E_real': e_true, 'E_predicha': e_pred,
                                       'Residuo': e_true-e_pred})

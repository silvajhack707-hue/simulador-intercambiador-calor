import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from modelo import (VARIABLES, N_MUESTRAS, generar_dataset, entrenar_modelo,
                    resolver_biseccion, resolver_epsilon_ntu)

st.set_page_config(page_title='Simulador de intercambiador de calor',
                   page_icon='🌡️', layout='wide')
st.title('Simulador de intercambiador de calor')
st.caption('Intercambiador de doble tubo en contracorriente · Bisección · ε-NTU · Random Forest')

@st.cache_resource(show_spinner='Generando 5000 escenarios y entrenando Random Forest...')
def preparar():
    df = generar_dataset()
    rf, metricas, predicciones = entrenar_modelo(df)
    return df, rf, metricas, predicciones

with st.sidebar:
    st.header('Navegación')
    pagina = st.radio('Sección', ['Simulación individual', 'Base de datos',
                                   'Evaluación de Random Forest', 'Convergencia de bisección'])
    st.divider()
    st.info('Modelo teórico con datos sintéticos y propiedades constantes. No sustituye mediciones reales de planta.')

if pagina == 'Simulación individual':
    st.subheader('Condiciones de operación')
    with st.form('datos'):
        c1,c2 = st.columns(2)
        with c1:
            mc = st.number_input('Caudal frío (kg/s)', min_value=0.01, value=1.50, step=0.1)
            tc = st.number_input('Temperatura de entrada fría (°C)', value=25.0, step=1.0)
            U = st.number_input('Coeficiente U (W/m²·K)', min_value=0.01, value=600.0, step=10.0)
            horas = st.number_input('Tiempo de operación (h)', min_value=0.01, value=1.0, step=0.5)
        with c2:
            mh = st.number_input('Caudal caliente (kg/s)', min_value=0.01, value=2.00, step=0.1)
            th = st.number_input('Temperatura de entrada caliente (°C)', value=100.0, step=1.0)
            A = st.number_input('Área de intercambio (m²)', min_value=0.01, value=10.0, step=1.0)
            eta = st.number_input('Eficiencia equivalente η', min_value=0.01, max_value=1.0, value=0.85, step=0.01)
        calcular = st.form_submit_button('Calcular', type='primary')
    if calcular:
        if th <= tc:
            st.error('La temperatura de entrada caliente debe superar la de entrada fría.')
        else:
            q, tco, tho, it, hist = resolver_biseccion(mc, mh, tc, th, U, A, guardar_historial=True)
            q_ntu = resolver_epsilon_ntu(mc, mh, tc, th, U, A)/1000
            kw = q/1000
            fila = pd.DataFrame([[mc,mh,tc,th,U,A]], columns=VARIABLES)
            dentro = (0.5 <= mc <= 3 and 0.5 <= mh <= 3 and 20 <= tc <= 50
                      and 70 <= th <= 140 and 300 <= U <= 1000 and 5 <= A <= 25)
            st.markdown('### Resultados físico-numéricos')
            a1,a2,a3 = st.columns(3)
            a1.metric('Potencia por bisección', f'{kw:.3f} kW')
            a2.metric('Potencia por ε-NTU', f'{q_ntu:.3f} kW')
            a3.metric('Consumo equivalente', f'{kw*horas/eta:.3f} kWh')
            st.write(f'**Temperaturas de salida:** fría = {tco:.3f} °C; caliente = {tho:.3f} °C. **Iteraciones:** {it}.')
            st.caption('El consumo incluye una eficiencia equivalente; no representa una medición real de una caldera.')
            if dentro:
                with st.spinner('Preparando modelo predictivo...'):
                    _, rf, _, _ = preparar()
                q_rf = float(rf.predict(fila)[0])
                e_rf = q_rf*horas/eta
                b1,b2 = st.columns(2)
                b1.metric('Potencia estimada por Random Forest', f'{q_rf:.3f} kW')
                b2.metric('Consumo estimado por Random Forest', f'{e_rf:.3f} kWh')
                st.caption(f'Diferencia relativa en potencia frente a bisección: {abs(q_rf-kw)/kw*100:.2f} %.')
            else:
                st.warning('Los valores están fuera del dominio de entrenamiento. Se muestran los resultados físicos, pero se omite Random Forest para evitar extrapolaciones no validadas.')
            st.markdown('#### Convergencia del cálculo')
            fig, ax = plt.subplots(figsize=(8,3.5))
            ax.semilogy(hist['Iteración'], np.maximum(abs(hist['Residuo (W)']), 1e-15), marker='.', markersize=4)
            ax.set(xlabel='Iteración', ylabel='|f(Q)| (W)')
            ax.grid(True, alpha=.3)
            st.pyplot(fig)
            plt.close(fig)
            st.download_button('Descargar historial CSV', hist.to_csv(index=False).encode('utf-8-sig'),
                               file_name='historial_biseccion.csv', mime='text/csv')

elif pagina == 'Base de datos':
    st.subheader('Escenarios sintéticos')
    with st.spinner('Generando escenarios y entrenando el modelo...'):
        df, _, _, _ = preparar()
    st.metric('Escenarios generados', f'{len(df):,}')
    st.dataframe(df, use_container_width=True, height=390)
    st.download_button('Descargar CSV', df.to_csv(index=False).encode('utf-8-sig'),
                       file_name='resultados_simulacion.csv', mime='text/csv')
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Escenarios')
    st.download_button('Descargar Excel', buffer.getvalue(),
                       file_name='resultados_simulacion.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.write(f'Error relativo máximo bisección vs ε-NTU: **{df["error_pct"].max():.10f} %**')
    st.write(f'Iteraciones promedio: **{df["bisection_iters"].mean():.2f}**')

elif pagina == 'Evaluación de Random Forest':
    st.subheader('Desempeño predictivo en los datos de prueba')
    with st.spinner('Preparando evaluación...'):
        _, rf, met, pred = preparar()
    a,b,c,d = st.columns(4)
    a.metric('R² potencia', f'{met["Q_R2"]:.5f}')
    b.metric('R² consumo', f'{met["E_R2"]:.5f}')
    c.metric('MAPE consumo', f'{met["E_MAPE"]:.3f} %')
    d.metric('MAE consumo', f'{met["E_MAE"]:.3f} kWh')
    st.caption(f'RMSE potencia: {met["Q_RMSE"]:.3f} kW; MAE potencia: {met["Q_MAE"]:.3f} kW; RMSE consumo: {met["E_RMSE"]:.3f} kWh.')
    fig, ax = plt.subplots(figsize=(7,4))
    ax.scatter(pred['E_real'], pred['E_predicha'], s=13, alpha=.4)
    extremo = max(pred['E_real'].max(), pred['E_predicha'].max())
    ax.plot([0,extremo],[0,extremo], color='red', label='Predicción ideal')
    ax.set(xlabel='Consumo físico-numérico (kWh)', ylabel='Consumo estimado por IA (kWh)',
           title='Consumo real del modelo vs. predicho')
    ax.grid(True, alpha=.25)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.scatter(pred['E_predicha'], pred['Residuo'], s=13, alpha=.4)
    ax.axhline(0, color='red')
    ax.set(xlabel='Consumo predicho (kWh)', ylabel='Residuo (kWh)', title='Residuos del consumo')
    ax.grid(True, alpha=.25)
    st.pyplot(fig)
    plt.close(fig)
    imp = pd.Series(rf.feature_importances_, index=VARIABLES).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(imp.index, imp.values)
    ax.set(xlabel='Variables de entrada', ylabel='Importancia relativa',
           title='Importancia de variables en Random Forest')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=.25)
    st.pyplot(fig)
    plt.close(fig)
    st.caption('Las importancias son predictivas, no porcentajes de causalidad física.')

else:
    st.subheader('Caso de comprobación de bisección')
    st.write('Parámetros: m_c = 1.5 kg/s, m_h = 2.0 kg/s, T_c,in = 25 °C, T_h,in = 100 °C, U = 600 W/(m²·K), A = 10 m².')
    q,tco,tho,n,hist = resolver_biseccion(1.5,2.0,25,100,600,10,guardar_historial=True)
    q_ntu = resolver_epsilon_ntu(1.5,2.0,25,100,600,10)
    a,b,c = st.columns(3)
    a.metric('Bisección', f'{q/1000:.6f} kW')
    b.metric('ε-NTU', f'{q_ntu/1000:.6f} kW')
    c.metric('Iteraciones', str(n))
    fig, ax = plt.subplots(figsize=(8,4))
    ax.semilogy(hist['Iteración'], np.maximum(abs(hist['Residuo (W)']),1e-15), marker='.', markersize=5)
    ax.set(xlabel='Iteración', ylabel='|f(Q)| (W)', title='Convergencia de bisección')
    ax.grid(True, alpha=.3)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(hist, use_container_width=True)
    st.download_button('Descargar historial', hist.to_csv(index=False).encode('utf-8-sig'),
                       file_name='historial_caso.csv', mime='text/csv')

st.divider()
st.caption('Proyecto académico · Resultados teóricos basados en un modelo idealizado de intercambiador de calor.')

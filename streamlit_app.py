import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from datetime import datetime, timedelta
from itertools import product
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class OptimizedSARIMAForecaster:
    def __init__(self, data):
        self.data = pd.Series(data)
        self.best_model = None
        self.best_params = None
        self.best_error = float('inf')
        
    def optimize_parameters(self, progress_bar=None):
        """
        Optimiza los parámetros SARIMA buscando el menor error
        """
        # Definir rangos de parámetros con decimales
        p_range = d_range = q_range = [round(x/100, 2) for x in range(11, 100, 11)]
        P_range = D_range = Q_range = [round(x/100, 2) for x in range(11, 100, 11)]
        
        # Crear todas las combinaciones posibles
        combinations = list(product(
            p_range[:3],  # Limitamos para mejor rendimiento
            d_range[:3],
            q_range[:3],
            P_range[:2],
            D_range[:2],
            Q_range[:2]
        ))
        
        total_combinations = len(combinations)
        
        for i, (p, d, q, P, D, Q) in enumerate(combinations):
            try:
                # Actualizar barra de progreso
                if progress_bar is not None:
                    progress_bar.progress(i / total_combinations)
                
                model = SARIMAX(
                    self.data,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                fitted_model = model.fit(disp=False)
                
                # Calcular error usando MAPE y RMSE combinados
                predictions = fitted_model.get_prediction(start=0)
                y_pred = predictions.predicted_mean
                
                mape = mean_absolute_percentage_error(self.data, y_pred)
                rmse = np.sqrt(mean_squared_error(self.data, y_pred))
                
                # Error combinado normalizado
                combined_error = (mape + rmse/np.mean(self.data))/2
                
                if combined_error < self.best_error:
                    self.best_error = combined_error
                    self.best_model = fitted_model
                    self.best_params = {
                        'p': p, 'd': d, 'q': q,
                        'P': P, 'D': D, 'Q': Q
                    }
                    
            except:
                continue
                
        return self.best_model, self.best_params, self.best_error
    
    def forecast(self, periods=12):
        """
        Genera pronósticos usando el mejor modelo
        """
        if self.best_model is None:
            raise ValueError("Debe optimizar los parámetros primero")
            
        forecast = self.best_model.forecast(periods)
        conf_int = self.best_model.get_forecast(periods).conf_int()
        return forecast, conf_int

# Configuración de la página
st.set_page_config(
    page_title="Pronóstico SARIMA Optimizado",
    page_icon="📊",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    </style>
""", unsafe_allow_html=True)

# Título y descripción
st.title("📊 Pronóstico SARIMA con Optimización Automática")
st.markdown("""
Esta aplicación realiza pronósticos utilizando el modelo SARIMA con optimización automática 
de parámetros en rangos decimales (0.11 a 0.99).

### Características:
- Optimización automática de parámetros estacionales y no estacionales
- Búsqueda del menor error promedio combinado (MAPE + RMSE)
- Visualización interactiva de resultados
- Intervalos de confianza personalizables
""")

# Input de datos
st.header("📝 Datos de Entrada")
data_input = st.text_area(
    "Ingresa tus datos históricos (un número por línea):",
    """100
120
140
160
130
110
105
125
145
165
135
115
110
130
150
170
140
120""",
    height=200
)

# Configuración
col1, col2 = st.columns(2)
with col1:
    periods = st.number_input(
        "Períodos a pronosticar",
        min_value=1,
        max_value=24,
        value=6
    )
    fecha_inicio = st.date_input(
        "Fecha de inicio",
        datetime.now() - timedelta(days=365)
    )

with col2:
    intervalo_confianza = st.slider(
        "Nivel de confianza (%)",
        min_value=80,
        max_value=99,
        value=95,
        step=1
    )
    frecuencia = st.selectbox(
        "Frecuencia de datos",
        ["Mensual", "Trimestral", "Anual"],
        index=0
    )

if st.button("🎯 Optimizar y Generar Pronóstico"):
    try:
        with st.spinner('Optimizando parámetros y generando pronóstico...'):
            # Convertir datos
            historical_data = [float(x) for x in data_input.strip().split('\n')]
            
            # Crear fechas
            if frecuencia == "Mensual":
                fechas = pd.date_range(fecha_inicio, periods=len(historical_data), freq='M')
            elif frecuencia == "Trimestral":
                fechas = pd.date_range(fecha_inicio, periods=len(historical_data), freq='Q')
            else:
                fechas = pd.date_range(fecha_inicio, periods=len(historical_data), freq='Y')
            
            # Crear y optimizar modelo
            forecaster = OptimizedSARIMAForecaster(historical_data)
            
            # Barra de progreso para optimización
            progress_bar = st.progress(0)
            st.info("Optimizando parámetros... Este proceso puede tomar varios minutos.")
            
            best_model, best_params, best_error = forecaster.optimize_parameters(progress_bar)
            progress_bar.empty()
            
            # Generar pronóstico
            forecast, conf_int = forecaster.forecast(periods)
            
            # Fechas para pronóstico
            fechas_forecast = pd.date_range(
                fechas[-1] + pd.Timedelta(days=1),
                periods=periods,
                freq=fechas.freq
            )
            
            # Mostrar resultados
            st.success("¡Optimización completada! 🎉")
            
            # Métricas principales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Error Combinado", f"{best_error:.4f}")
            with col2:
                params_str = ", ".join([f"{k}={v:.2f}" for k, v in best_params.items()])
                st.metric("Mejores Parámetros", params_str)
            with col3:
                st.metric("Períodos Pronosticados", periods)
            
            # Tabs para diferentes vistas
            tab1, tab2, tab3 = st.tabs(["📈 Gráfico", "📊 Datos", "📑 Estadísticas"])
            
            with tab1:
                fig = go.Figure()
                
                # Datos históricos
                fig.add_trace(go.Scatter(
                    x=fechas,
                    y=historical_data,
                    name="Datos Históricos",
                    line=dict(color='#1f77b4')
                ))
                
                # Pronóstico
                fig.add_trace(go.Scatter(
                    x=fechas_forecast,
                    y=forecast,
                    name="Pronóstico",
                    line=dict(color='#2ca02c')
                ))
                
                # Intervalo de confianza
                fig.add_trace(go.Scatter(
                    x=fechas_forecast.tolist() + fechas_forecast.tolist()[::-1],
                    y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(44, 160, 44, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Intervalo de Confianza ({intervalo_confianza}%)'
                ))
                
                fig.update_layout(
                    title="Pronóstico con Parámetros Optimizados",
                    xaxis_title="Fecha",
                    yaxis_title="Valor",
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Datos Históricos")
                    hist_df = pd.DataFrame({
                        'Fecha': fechas,
                        'Valor': historical_data
                    })
                    st.dataframe(hist_df, hide_index=True)
                
                with col2:
                    st.subheader("Pronóstico")
                    forecast_df = pd.DataFrame({
                        'Fecha': fechas_forecast,
                        'Pronóstico': forecast.values,
                        'Límite Inferior': conf_int.iloc[:, 0],
                        'Límite Superior': conf_int.iloc[:, 1]
                    })
                    st.dataframe(forecast_df, hide_index=True)
            
            with tab3:
                st.subheader("Detalles del Modelo Optimizado")
                
                # Parámetros optimizados
                st.write("#### Parámetros SARIMA Optimizados")
                params_df = pd.DataFrame([best_params])
                st.dataframe(params_df)
                
                # Métricas de error
                st.write("#### Métricas de Error")
                predictions = best_model.get_prediction(start=0)
                y_pred = predictions.predicted_mean
                
                error_metrics = {
                    'MAPE': mean_absolute_percentage_error(historical_data, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(historical_data, y_pred)),
                    'AIC': best_model.aic,
                    'BIC': best_model.bic
                }
                
                error_df = pd.DataFrame([error_metrics])
                st.dataframe(error_df)
                
                # Resumen del modelo
                st.write("#### Resumen Estadístico")
                st.code(str(best_model.summary()))
        
    except Exception as e:
        st.error(f"Error en el procesamiento: {str(e)}")
        st.info("Verifica que los datos ingresados sean números válidos y que haya suficientes observaciones.")
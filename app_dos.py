import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import seaborn as sns
from itertools import product
import base64
import streamlit.components.v1 as components

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
# Suprimir advertencias ValueWarning
warnings.simplefilter("ignore")



# Configuración de Streamlit
st.set_page_config(page_title="Crypto Modelo Predicción", page_icon="img/icono-page.png", layout="wide")

theme_plotly = None

#"""" codigo de particulas que se agregan en le background""""
particles_js = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Particles.js</title>
  <style>
  #particles-js {
    background-color: #191970;    
    position: fixed;
    width: 100vw;
    height: 100vh;
    top: 0;
    left: 0;
    z-index: -1; /* Send the animation to the back */
  }
  .content {
    position: relative;
    z-index: 1;
    color: white;
  }
  
</style>
</head>
<body>
  <div id="particles-js"></div>
  <div class="content">
    <!-- Placeholder for Streamlit content -->
  </div>
  <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
  <script>
    particlesJS("particles-js", {
      "particles": {
        "number": {
          "value": 300,
          "density": {
            "enable": true,
            "value_area": 800
          }
        },
        "color": {
          "value": "#fffc33"
        },
        "shape": {
          "type": "circle",
          "stroke": {
            "width": 0,
            "color": "#000000"
          },
          "polygon": {
            "nb_sides": 5
          },
          "image": {
            "src": "img/github.svg",
            "width": 100,
            "height": 100
          }
        },
        "opacity": {
          "value": 0.5,
          "random": false,
          "anim": {
            "enable": false,
            "speed": 1,
            "opacity_min": 0.2,
            "sync": false
          }
        },
        "size": {
          "value": 2,
          "random": true,
          "anim": {
            "enable": false,
            "speed": 40,
            "size_min": 0.1,
            "sync": false
          }
        },
        "line_linked": {
          "enable": true,
          "distance": 100,
          "color": "#fffc33",
          "opacity": 0.22,
          "width": 1
        },
        "move": {
          "enable": true,
          "speed": 0.2,
          "direction": "none",
          "random": false,
          "straight": false,
          "out_mode": "out",
          "bounce": true,
          "attract": {
            "enable": false,
            "rotateX": 600,
            "rotateY": 1200
          }
        }
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": {
          "onhover": {
            "enable": true,
            "mode": "grab"
          },
          "onclick": {
            "enable": true,
            "mode": "repulse"
          },
          "resize": true
        },
        "modes": {
          "grab": {
            "distance": 100,
            "line_linked": {
              "opacity": 1
            }
          },
          "bubble": {
            "distance": 400,
            "size": 2,
            "duration": 2,
            "opacity": 0.5,
            "speed": 1
          },
          "repulse": {
            "distance": 200,
            "duration": 0.4
          },
          "push": {
            "particles_nb": 2
          },
          "remove": {
            "particles_nb": 3
          }
        }
      },
      "retina_detect": true
    });
  </script>
</body>
</html>
"""
globe_js = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vanta Globe Animation</title>
    <style type="text/css">
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }
      body {
        overflow: hidden;
        height: 100%;
        margin: 0;
        background-color: #1817ed; /* Fondo azul */
      }
      #canvas-globe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
      }
    </style>
  </head>
  <body>
    <div id="canvas-globe"></div>       

    <!-- Scripts de Three.js y Vanta.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vanta/0.5.24/vanta.globe.min.js"></script>

    <script type="text/javascript">      
      document.addEventListener("DOMContentLoaded", function() {
        VANTA.GLOBE({
          el: "#canvas-globe", // El elemento donde se renderiza la animación
          mouseControls: true,
          touchControls: true,
          gyroControls: false,
          minHeight: 200.00,
          minWidth: 200.00,
          scale: 1.00,
          scaleMobile: 1.00,
          color: 0xd1ff3f, // Color verde amarillento
          backgroundColor: 0x1817ed // Fondo azul
        });
      });
    </script>
  </body>
</html>
"""

waves_js = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vanta Waves Animation</title>
    <style type="text/css">
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }
      html, body {
        height: 100%;
        margin: 0;
        overflow: hidden;
      }
      #canvas-dots {
        position: absolute;
        width: 100%;
        height: 100%;
      }
    </style>
  </head>
  <body>
    <div id="canvas-waves"></div>       
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vanta/0.5.24/vanta.waves.min.js"></script>
    
    <script type="text/javascript">      
      document.addEventListener("DOMContentLoaded", function() {
        VANTA.WAVES({
          el: "#canvas-waves", // Especificar el contenedor donde debe renderizarse
           mouseControls: true,
           touchControls: true,
           gyroControls: false,
           minHeight: 200.00,
           minWidth: 200.00,
           scale: 1.00,
           scaleMobile: 1.00,
           color: 0x15159b
        });
      });
    </script>
  </body>
</html>
"""

#""" imagen de background"""
def add_local_background_image(image):
  with open(image, "rb") as image:
    encoded_string = base64.b64encode(image.read())
    st.markdown(
      f"""
      <style>
      .stApp{{
        background-image: url(data:files/{"jpg"};base64,{encoded_string.decode()});
      }}    
      </style>
      """,
      unsafe_allow_html=True
    )
add_local_background_image("img/fondo.jpg")

#""" imagen de sidebar"""
def add_local_sidebar_image(image):
  with open(image, "rb") as image:
    encoded_string = base64.b64encode(image.read())
    st.markdown(
      f"""
      <style>
      .stSidebar{{
        background-image: url(data:files/{"jpg"};base64,{encoded_string.decode()});
      }}    
      </style>
      """,
      unsafe_allow_html=True
    )

add_local_sidebar_image("img/fondo1.jpg")

# Agregar imágenes
# ---- animación de inicio de pagina----
with st.container():
    #st.write("---")
    left, right = st.columns(2, gap='small', vertical_alignment="center")
    with left:
        components.html(waves_js, height=150,scrolling=False)
    with right:
       components.html(particles_js, height=150,scrolling=False) 
    #st.write("---")    
#-------------- animacion con css de los botones  ------------------------
with open('asset/styles.css') as f:
        css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Título principal
st.title("Modelo de Pronóstico ARIMA para Criptomonedas")
st.write('---')

# Parámetros de la barra lateral
with st.sidebar:
    st.header("Parámetros del Modelo")
    crypto_symbol = st.text_input("Símbolo de la Criptomoneda", "BTC-USD", help="Ingrese el símbolo en mayúsculas, por ejemplo, BTC-USD")
    crypto_symbol = crypto_symbol.upper()
    periodo = st.number_input("Meses de datos históricos", min_value=1, max_value=12, value=6, step=1)
    prediction_ahead = st.number_input("Días a predecir", min_value=1, max_value=30, value=15, step=1)
    st.write('---')
     #predecir = st.button("Predecir")

if st.sidebar.button("Predecir", key="predecir", use_container_width=True): #if predecir:
    # Paso 1: Obtener datos de la criptomoneda
    df_data = yf.download(crypto_symbol, period=f'{periodo}mo', interval='1d', auto_adjust=False)
    df_data = df_data[['Close']].dropna()

    if df_data.empty:
        st.error("No se pudieron obtener datos para el símbolo ingresado. Verifique el símbolo e intente nuevamente.")
    else:
        # División de datos en entrenamiento y prueba
        train_size = int(len(df_data) * 0.8)
        train, test = df_data[:train_size], df_data[train_size:]

        # Paso 2: Ajustar el modelo ARIMA
        p_values = range(0, 4)
        d_values = range(0, 2)
        q_values = range(0, 4)

        def evaluate_arima_model(train, test, arima_order):
            try:
                model = ARIMA(train, order=arima_order)
                model_fit = model.fit()
                predictions = model_fit.forecast(steps=len(test))
                mse = mean_squared_error(test, predictions)
                return mse, model_fit
            except:
                return float('inf'), None

        results = []
        for p, d, q in product(p_values, d_values, q_values):
            order = (p, d, q)
            mse, model_fit = evaluate_arima_model(train['Close'], test['Close'], order)
            results.append((order, mse, model_fit))

        # Seleccionar el mejor modelo
        best_order, best_mse, best_model = min(results, key=lambda x: x[1])

        if best_model is None:
            st.error("No se pudo encontrar un modelo ARIMA válido. Por favor, ajuste los parámetros.")
        else:
            # Paso 3: Realizar la predicción
            forecast = best_model.forecast(steps=len(test) + prediction_ahead)
            latest_close_price = float(df_data['Close'].iloc[-1])
            last_predicted_price = float(forecast[-1])

            # Mostrar métricas
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"Precio de Cierre Actual ({crypto_symbol})", value=f"${latest_close_price:,.2f}")
            with col2:
                st.metric(label=f"Precio Proyectado a {prediction_ahead} Días", value=f"${last_predicted_price:,.2f}")

            # Gráfico de predicciones
            plt.figure(figsize=(14, 5))
            plt.plot(df_data.index, df_data['Close'], label='Actual', color='blue')
            plt.axvline(x=df_data.index[train_size], color='gray', linestyle='--', label='División Entrenamiento/Prueba')
            plt.plot(test.index, forecast[:len(test)], label='Predicciones de Prueba', color='orange')
            future_index = pd.date_range(start=test.index[-1], periods=prediction_ahead + 1, freq='D')[1:]
            plt.plot(future_index, forecast[len(test):], label='Predicciones Futuras', color='red')
            plt.title(f'Predicciones del Modelo ARIMA para {crypto_symbol}')
            plt.xlabel('Fecha')
            plt.ylabel('Precio (USD)')
            plt.legend()
            st.pyplot(plt)

            # Cálculo de rendimiento y volatilidad diaria
            df_data['Rendimiento_Diario'] = df_data['Close'].pct_change() * 100
            df_data['Volatilidad_Diaria'] = df_data['Rendimiento_Diario'].rolling(window=7).std()

            # Eliminar filas con valores NaN o infinitos
            df_data_clean = df_data.replace([np.inf, -np.inf], np.nan).dropna()

            # Gráfico de regresión lineal entre volatilidad y rendimiento
            if not df_data_clean.empty:
                X = df_data_clean['Volatilidad_Diaria'].values.reshape(-1, 1)
                y = df_data_clean['Rendimiento_Diario'].values

                reg = LinearRegression().fit(X, y)
                y_pred = reg.predict(X)

                plt.figure(figsize=(14, 5))
                plt.scatter(X, y, color='blue', alpha=0.6, label='Datos Reales')
                plt.plot(X, y_pred, color='red', linewidth=2, label='Línea de Regresión')
                plt.title(f'Relación entre Volatilidad y Rendimiento Diario - {crypto_symbol}')
                plt.xlabel('Volatilidad Diaria (%)')
                plt.ylabel('Rendimiento Diario (%)')
                plt.legend()
                st.pyplot(plt)
            else:
                st.warning("No hay suficientes datos válidos para realizar el análisis de regresión.")

            # Histogramas de rendimiento y volatilidad diaria
            col1, col2 = st.columns(2)
            with col1:
                plt.figure(figsize=(7, 4))
                plt.hist(df_data_clean['Rendimiento_Diario'], bins=50, color='blue', alpha=0.7)
                plt.title(f'Histograma de Rendimiento Diario - {crypto_symbol}')
                plt.xlabel('Rendimiento Diario (%)')
                plt.ylabel('Frecuencia')
                st.pyplot(plt)
            with col2:
                plt.figure(figsize=(7, 4))
                plt.hist(df_data_clean['Volatilidad_Diaria'], bins=50, color='red', alpha=0.7)
                plt.title(f'Histograma de Volatilidad Diaria - {crypto_symbol}')
                plt.xlabel('Volatilidad Diaria (%)')
                plt.ylabel('Frecuencia')
                st.pyplot(plt)

            # Matriz de regresión y estimación de densidad
            sns.set(style="whitegrid")
            pairplot_fig = sns.pairplot(df_data_clean[['Rendimiento_Diario', 'Volatilidad_Diaria']], kind='reg', diag_kind='kde')
            st.pyplot(pairplot_fig)
# --------------- footer -----------------------------
st.write("---")
with st.container():
  #st.write("---")
  st.write("&copy; - derechos reservados -  2024 -  Walter Gómez - FullStack Developer - Data Science - Business Intelligence")
  #st.write("##")
  left, right = st.columns(2, gap='medium', vertical_alignment="bottom")
  with left:
    #st.write('##')
    st.link_button("Mi LinkedIn", "https://www.linkedin.com/in/walter-gomez-fullstack-developer-datascience-businessintelligence-finanzas-python/",use_container_width=True)
  with right: 
     #st.write('##') 
    st.link_button("Mi Porfolio", "https://walter-portfolio-animado.netlify.app/", use_container_width=True)
      
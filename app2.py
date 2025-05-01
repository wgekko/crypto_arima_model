import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
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

# --- Título principal ---
st.button("Modelo de pronóstico ARIMA", key="pulse", use_container_width=True)
st.write('###')

# --- Barra lateral de parámetros ---
with st.sidebar:
    components.html(globe_js, height=150,scrolling=False)
    st.button("Parámetros del modelo", key="topulse", use_container_width=True)
    crypto_symbol = st.sidebar.text_input("Símbolo Cryptomoneda", "BTC-USD", help="Debe ingresar el dato en mayúsculas").upper()
    periodo = st.sidebar.number_input("Meses históricos para calcular", min_value=1, max_value=12, value=6, step=1, help="Valor debe oscilar entre 1 y 12")
    prediction_ahead = st.sidebar.number_input("Días a Predecir Precio", min_value=1, max_value=30, value=15, step=1, help="Valor debe oscilar entre 1 y 30")
    st.write("###")

# --- Lógica principal de predicción ---
if st.sidebar.button("Predecir", key="predecir", use_container_width=True):
    with st.spinner(f"Descargando datos de {crypto_symbol} y entrenando el modelo..."):
        # Paso 1: Obtener datos de criptomonedas
        try:
            df_data = yf.download(crypto_symbol, period=f'{periodo}mo', interval='1d', auto_adjust=False)
            df_data = df_data[['Close']].dropna()
        except Exception as e:
            st.error(f"Error al descargar datos de {crypto_symbol}: {e}")
            st.stop()

        # Preparar la división de entrenamiento y prueba
        train_size = int(len(df_data) * 0.8)
        train, test = df_data[:train_size], df_data[train_size:]

        # Paso 2: Ajustar el modelo ARIMA (Optimización)
        p_values = range(0, 3)  # Reducir el espacio de búsqueda para eficiencia
        d_values = range(0, 2)
        q_values = range(0, 3)

        best_mse = float('inf')
        best_order = None
        best_model_fit = None

        for order in product(p_values, d_values, q_values):
            try:
                model = ARIMA(train['Close'], order=order)
                model_fit = model.fit()
                predictions = model_fit.forecast(steps=len(test))
                mse = mean_squared_error(test['Close'], predictions)
                if mse < best_mse:
                    best_mse = mse
                    best_order = order
                    best_model_fit = model_fit
            except Exception as e:
                print(f"Error al ajustar ARIMA con parámetros {order}: {e}")

        if best_model_fit is None:
            st.error("No se pudo encontrar un modelo ARIMA válido con los parámetros probados.")
        else:
            # Paso 3: Realizar la predicción
            forecast = best_model_fit.forecast(steps=len(test) + prediction_ahead)
            latest_close_price = float(df_data['Close'].iloc[-1])
            last_predicted_price = float(forecast[-1])

            # Mostrar métricas centradas
            col1, col2 = st.columns(2, border=True, vertical_alignment="center")
            with col1:
                st.subheader(f"Precio de Cierre de : {crypto_symbol}")
                st.button(f" -- U$D {latest_close_price:,.2f} --", key="inpulse")
            with col2:
                st.subheader(f"Precio proyectado a {prediction_ahead} Día/s")
                st.button(f" -- U$D {last_predicted_price:,.2f} --", key="toinpulse")

            # Graficar resultados
            plt.figure(figsize=(14, 5)) # Ajuste de altura
            plt.plot(df_data.index, df_data['Close'], label='Actual', color='blue')
            plt.axvline(x=df_data.index[train_size], color='gray', linestyle='--', label='Train/Test Split')
            plt.plot(train.index, train['Close'], label='Train Data', color='green')
            plt.plot(test.index, forecast[:len(test)], label='Test Predictions', color='orange')
            future_index = pd.date_range(start=test.index[-1], periods=prediction_ahead + 1, freq='D')[1:]
            plt.plot(future_index, forecast[len(test):], label=f'{prediction_ahead}-Day Forecast', color='red')
            plt.title(f'{crypto_symbol} Predicciones del modelo ARIMA (Mejor orden: {best_order})')
            plt.xlabel('Días')
            plt.ylabel('Precio (USD)')
            plt.legend()
            with st.container(border=True):
                st.subheader(f"Predicción del Modelo ARIMA para : {crypto_symbol}")
                st.pyplot(plt)

            st.write("---")
            with st.container(border=True):
                left_col, right_col = st.columns(2, gap='small', vertical_alignment="center")
                with left_col:
                    # Histograma de rendimiento diario
                    df_data['Rendimiento_Diario'] = df_data['Close'].pct_change() * 100
                    plt.figure(figsize=(14, 5)) # Ajuste de altura
                    sns.histplot(df_data['Rendimiento_Diario'].dropna(), bins=50, color='blue', alpha=0.7)
                    plt.title(f'Histograma de Rendimiento Diario de {crypto_symbol}')
                    plt.xlabel('Rendimiento Diario (%)')
                    plt.ylabel('Frecuencia')
                    st.markdown(f"#### Histograma de Rendimiento Diario de {crypto_symbol}")
                    st.pyplot(plt)

                with right_col:
                    # Histograma de volatilidad diaria
                    df_data['Volatilidad_Diaria'] = df_data['Rendimiento_Diario'].rolling(window=7).std()
                    plt.figure(figsize=(14, 5)) # Ajuste de altura
                    sns.histplot(df_data['Volatilidad_Diaria'].dropna(), bins=50, color='red', alpha=0.7)
                    plt.title(f'Histograma de Volatilidad Diaria de {crypto_symbol}')
                    plt.xlabel('Volatilidad Diaria (%)')
                    plt.ylabel('Frecuencia')
                    st.markdown(f"#### Histograma de Volatilidad Diaria de {crypto_symbol}")
                    st.pyplot(plt)

            # Reemplazar NaN con 0 de forma más eficiente
            df_data.fillna(0, inplace=True)

            # Gráfico de rendimiento vs volatilidad con regresión lineal
            X = df_data['Volatilidad_Diaria'].values.reshape(-1, 1)
            y = df_data['Rendimiento_Diario'].values
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)

            plt.figure(figsize=(14, 5)) # Ajuste de altura
            plt.scatter(df_data['Volatilidad_Diaria'], df_data['Rendimiento_Diario'], color='blue', alpha=0.7, label='Datos reales')
            plt.plot(df_data['Volatilidad_Diaria'], y_pred, color='red', linewidth=2, label='Recta de regresión')
            plt.title(f'Relación entre Volatilidad Diaria y Rendimiento Diario de {crypto_symbol}')
            plt.xlabel('Volatilidad Diaria (%)')
            plt.ylabel('Rendimiento Diario (%)')
            plt.legend()

            st.write("---")
            with st.container(border=True):
                st.subheader(f"Relación entre Volatilidad y Rendimiento de {crypto_symbol}")
                st.pyplot(plt)

            g = sns.pairplot(data=df_data[["Volatilidad_Diaria", "Rendimiento_Diario"]], kind="reg")
            g.fig.set_size_inches(14, 5) # Ajuste de altura
            st.pyplot(g)
            with st.container(border=True):
                st.subheader(f"Matriz diagonal de regresión de {crypto_symbol}")
                st.pyplot(plt) # ¡Cuidado! Aquí estás mostrando el último plt, no el de pairplot.

            g_kde = sns.pairplot(data=df_data[["Volatilidad_Diaria", "Rendimiento_Diario"]], diag_kind="kde")
            g_kde.fig.set_size_inches(14, 5) # Ajuste de altura
            st.pyplot(g_kde)
            with st.container(border=True):
                st.subheader(f"Matriz diagonal estimación de densidad de {crypto_symbol}")
                st.pyplot(plt) # ¡Cuidado! Mismo problema que antes.
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
      
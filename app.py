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

theme_plotly = None

# Configuración de Streamlit
st.set_page_config(page_title="Crypto Modelo Predicción", page_icon="img/icono-page.png", layout="wide")

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
st.button("Modelo de pronóstico ARIMA" , key="pulse", use_container_width=True)
st.write('###')
# Parámetros de la barra lateral
with st.sidebar:
    components.html(globe_js, height=150,scrolling=False)
    st.button("Parámetros del modelo" ,key="topulse" , use_container_width=True)
    crypto_symbol = st.sidebar.text_input("Simbolo Cryptomoneda", "BTC-USD" , help="debe ingresar el dato en Mayusculas")
    crypto_symbol = crypto_symbol.upper()
    #rypto_symbol = st.selectbox("Simbolo Cryptomoneda",())
    periodo = st.sidebar.number_input("Meses historico para calcular", min_value=1, max_value=12, value=6, step=1, help="valor debe oscilar entre 1/12")
    prediction_ahead = st.sidebar.number_input("Días a Predecir Precio", min_value=1, max_value=30, value=15, step=1, help="valor debe oscilar entre 1/30")
    st.write("###")
if st.sidebar.button("Predecir", key="predecir", use_container_width=True):
    # Paso 1: Obtener datos de criptomonedas para los últimos 3 meses
    df_data = yf.download(crypto_symbol, period=f'{periodo}mo', interval='1d')
    df_data = df_data[['Close']].dropna()
    # Preparar la división de entrenamiento y prueba (80% para entrenamiento, 20% para prueba)
    train_size = int(len(df_data) * 0.8)
    train, test = df_data[:train_size], df_data[train_size:]
    # Paso 2: Ajustar el modelo ARIMA
    p_values = range(0, 4)  # Definir el rango para ARIMA(p,d,q)
    d_values = range(0, 2)
    q_values = range(0, 4)

    def evaluate_arima_model(train, test, arima_order):
        try:
            model = ARIMA(train, order=arima_order)
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
            mse = mean_squared_error(test, predictions)
            return mse, model_fit
        except Exception as e:
            print(f"Error al ajustar ARIMA con parámetros {arima_order}: {e}")
            return float('inf'), None

    results = []
    mse_values = []  # To store the MSE for each model combination
    arima_combinations = []  # Store the combinations of (p, d, q)
    for p, d, q in product(p_values, d_values, q_values):
        arima_order = (p, d, q)
        mse, model_fit = evaluate_arima_model(train['Close'], test['Close'], arima_order)
        results.append((arima_order, mse, model_fit))
        mse_values.append(mse)
        arima_combinations.append(f"({p},{d},{q})")

    # Seleccionar el mejor modelo
    best_order, best_mse, best_model = min(results, key=lambda x: x[1])

    # Comprobar si el modelo encontrado es válido
    if best_model is None:
        st.error("No se pudo encontrar un modelo ARIMA válido. Por favor, ajusta los parámetros.")
    else:
        # Paso 3: Realizar la predicción
        forecast = best_model.forecast(steps=len(test) + prediction_ahead)
        # Último precio de cierre y último precio predicho
        latest_close_price = float(df_data['Close'].iloc[-1])
        last_predicted_price = float(forecast[-1])
        # Diseño centrado para las métricas            
        col1, col2= st.columns(2, border=True, vertical_alignment="center")        
        st.markdown("<div style='display: flex; justify-content: center; align-items: center; height: 100%;'>", unsafe_allow_html=True)
        with col1:                       
            st.subheader(f"""Precio de Cierre de : {crypto_symbol}""")
            st.button(f" -- U$D {latest_close_price:,.2f}  --", key="inpulse") 
        with col2:
            st.subheader(f"""Precio proyectado a {prediction_ahead} Dia/s """)
            st.button(f" -- U$D  {last_predicted_price:,.2f}  --", key="toinpulse")     
        st.markdown("</div>", unsafe_allow_html=True) 
        
        # Graficar los resultados
        plt.figure(figsize=(14, 4))  # Ajuste de altura para que el gráfico sea más bajo
        plt.plot(df_data.index, df_data['Close'], label='Actual', color='blue')
        plt.axvline(x=df_data.index[train_size], color='gray', linestyle='--', label='Train/Test Split')

        # Datos de entrenamiento/prueba y predicciones
        plt.plot(train.index, train['Close'], label='Train Data', color='green')
        plt.plot(test.index, forecast[:len(test)], label='Test Predictions', color='orange')

        # Predicciones futuras
        future_index = pd.date_range(start=test.index[-1], periods=prediction_ahead + 1, freq='D')[1:]
        plt.plot(future_index, forecast[len(test):], label=f'{prediction_ahead}-Day Forecast', color='red')

        plt.title(f'{crypto_symbol} Predicciones del modelo ARIMA')
        plt.xlabel('Dias')
        plt.ylabel('Precio (USD)')
        plt.legend()
        #st.write("---")
        with st.container(border=True):          
          st.subheader(f"""Predicción del Modelo ARIMA para : {crypto_symbol} """)    
          st.pyplot(plt)       
        
    # Additional Plot: MSE vs ARIMA parameters
    #plt.figure(figsize=(14, 4))
    #plt.plot(arima_combinations, mse_values, marker='o', linestyle='-', color='purple')
    #plt.title('MSE para diferentes combinaciones de parámetros ARIMA')
    #plt.xlabel('ARIMA Parámetros (p, d, q)')
    #plt.ylabel('Error cuadrático medio (MSE)')
    #plt.xticks(rotation=45)
    
    #with st.container(border=True):
    #  st.subheader("Combinaciones de parámetros ARIMA/MSE")  
    #  st.pyplot(plt)
    st.write("---")
    with st.container(border=True):
    #st.write("---")       
      left, right = st.columns(2, gap='small',vertical_alignment="center")  
      # **Histograma de rendimiento diario**
      with left:
        df_data['Rendimiento_Diario'] = df_data['Close'].pct_change() * 100  
        #df_data['Rendimiento_Diario'] = df_data['Close'].pct_change() 
        plt.figure(figsize=(14, 4))
        plt.hist(df_data['Rendimiento_Diario'].dropna(), bins=50, color='blue', alpha=0.7)
        plt.title(f'Histograma de Rendimiento Diario de {crypto_symbol}')
        plt.xlabel('Rendimiento Diario (%)')
        plt.ylabel('Frecuencia')
        #st.write("---")
        with st.container(border=True):
                #st.subheader(f"Histograma de Rendimiento Diario de {crypto_symbol}") 
                st.markdown(f"#### Histograma de Rendimiento Diario de {crypto_symbol}") 
                st.pyplot(plt)
      with right:          

        # **Histograma de volatilidad diaria**
          # Calcular el rendimiento diario
        df_data['Volatilidad_Diaria'] = df_data['Rendimiento_Diario'].rolling(window=7).std()
        plt.figure(figsize=(14, 4))
        plt.hist(df_data['Volatilidad_Diaria'].dropna(), bins=50, color='red', alpha=0.7)
        plt.suptitle(f'Histograma de Volatilidad Diaria de {crypto_symbol}')
        plt.xlabel('Volatilidad Diaria (%)')
        plt.ylabel('Frecuencia')
        #st.write("---")
        with st.container(border=True):
            #st.subheader(f"Histograma de Volatilidad Diaria de {crypto_symbol}") 
            st.markdown(f"#### Histograma de Volatilidad Diaria de {crypto_symbol}") 
            st.pyplot(plt)      
          
    # Reemplazar explícitamente 'None' o 'NaN' con 0
    df_data['Rendimiento_Diario'] = df_data['Rendimiento_Diario'].replace([None, np.nan], 0)
    df_data['Volatilidad_Diaria'] = df_data['Volatilidad_Diaria'].replace([None, np.nan], 0)
    # Usar 'apply()' para reemplazar 'NaN' por 0
    df_data['Rendimiento_Diario'] = df_data['Rendimiento_Diario'].apply(lambda x: 0 if pd.isna(x) else x)
    df_data['Volatilidad_Diaria'] = df_data['Volatilidad_Diaria'].apply(lambda x: 0 if pd.isna(x) else x)
  
    # **Crear gráfico de rendimiento vs volatilidad con regresión lineal**
    X = df_data['Volatilidad_Diaria'].values.reshape(-1, 1)  # Volatilidad como variable independiente
    y = df_data['Rendimiento_Diario'].values  # Rendimiento como variable dependiente

    # Ajustar el modelo de regresión lineal
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)  # Predicciones de la regresión

    # Graficar los puntos y la línea de regresión
    plt.figure(figsize=(14, 4))
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
    
    #g =sns.jointplot(data=df_data, x='Volatilidad_Diaria', y= 'Rendimiento_Diario', kind="reg")
    g =sns.pairplot(data=df_data,vars = ["Volatilidad_Diaria", "Rendimiento_Diario"], kind="reg")
    g.fig.set_size_inches(14, 4)
    plt.show()
    with st.container(border=True):
        st.subheader(f"Matriz diagonal de regresión de {crypto_symbol}")
        st.pyplot(plt)     
        #g =sns.jointplot(data=df_data, x='Volatilidad_Diaria', y= 'Rendimiento_Diario', kind="reg")
    
    
    
    g =sns.pairplot(data=df_data, diag_kind="kde")
    g.fig.set_size_inches(14, 4)
    plt.show()    
    with st.container(border=True):
        st.subheader(f"Matriz diagonal estimación de densidad de {crypto_symbol}")
        st.pyplot(plt) 
      
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
      
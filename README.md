# 📊 Curso de Data Science - Plan de Estudios

## 🎯 Vista General

**Duración sugerida:** 16 semanas (8–10 h/semana). Incluyo variante intensiva (8 semanas) y extendida (24 semanas).

**Perfil de salida:** Capaz de tomar un problema de negocio, obtener/limpiar datos, explorar, modelar (ML clásico), evaluar, comunicar hallazgos y desplegar un modelo sencillo.

### 🛠️ Stack Principal
- **Python** (Anaconda/Miniconda)
- **Jupyter/VS Code**
- **Git/GitHub**
- **SQL**
- **scikit‑learn, pandas, matplotlib/plotly**

> **Opcional:** PyTorch o TensorFlow para un módulo de Deep Learning

## ⚙️ Requisitos y Puesta a Punto (Semana 0)

### 📚 Conocimientos Previos
- Aritmética básica
- Nociones de álgebra lineal (vectores/matrices)
- Estadística descriptiva elemental
- **No es obligatorio saber programar**

### 💻 Instalación Mínima (Windows/macOS/Linux)

```bash
# 1) Instalar Miniconda (o Anaconda)
# 2) Crear entorno
conda create -n ds python=3.11 -y
conda activate ds

# 3) Paquetes base
conda install -c conda-forge numpy pandas matplotlib plotly scikit-learn jupyterlab ipywidgets -y
pip install seaborn xgboost lightgbm shap statsmodels optuna category_encoders

# 4) (Opcional) Deep Learning
pip install torch torchvision torchaudio  # o: pip install tensorflow
```

### 🔧 Herramientas
- **VS Code** + extensiones (Python, Jupyter)
- **Git, GitHub**

### ✅ Buenas Prácticas desde el Día 1
- Control de versiones (un commit por hito)
- README claros
- Notebooks limpios

## 📅 Plan de 16 Semanas (Ruta Estándar)

> **Cada semana indica:** Objetivos, Contenido, Práctica, Entregable.

### 📚 Semana 1 — Fundamentos de Python para Data Science

**🎯 Objetivos:** tipos, estructuras de datos, control de flujo, funciones, módulos, virtualenv/conda, notebooks.

**📖 Contenido:** listas/tuplas/dict, list comprehensions, manejo de errores, lectura/escritura de archivos.

**💻 Práctica:** mini‑ejercicios (FizzBuzz, conteo de palabras, parsing CSV simple).

**📋 Entregable:** notebook con 8–10 ejercicios resueltos y testeados.

### 🔢 Semana 2 — Numpy y Vectorización

**🎯 Objetivos:** arrays, broadcasting, álgebra lineal básica, rendimiento vs. bucles.

**📖 Contenido:** ndarray, slicing, einsum, tiempos con %timeit.

**💻 Práctica:** operaciones matriciales, normalización/estandarización desde cero.

**📋 Entregable:** notebook comparando soluciones vectorizadas vs. bucles.

### 🐼 Semana 3 — Pandas y Data Wrangling

**🎯 Objetivos:** carga/limpieza/unión/transformación de datos tabulares.

**📖 Contenido:** DataFrame, joins, groupby, pivot, manejo de nulos, tipos de fecha.

**💻 Práctica:** pipeline de limpieza sobre 2 datasets (ventas + clientes).

**📋 Entregable:** script etl_basic.py y notebook EDA inicial.

### 📊 Semana 4 — Visualización de Datos

**🎯 Objetivos:** comunicar hallazgos con gráficos efectivos.

**📖 Contenido:** matplotlib, seaborn, plotly; buenas prácticas (títulos, ejes, anotaciones).

**💻 Práctica:** dashboard exploratorio simple (plotly) con 4–6 gráficos clave.

**📋 Entregable:** notebook + html exportado del dashboard.

### 📈 Semana 5 — Estadística Descriptiva e Inferencial I

**🎯 Objetivos:** distribuciones, muestreo, intervalos de confianza, pruebas A/B.

**📖 Contenido:** media/mediana/varianza, bootstrapping, test t, chi‑cuadrado.

**💻 Práctica:** simulaciones Monte Carlo para estimar un intervalo.

**📋 Entregable:** reporte corto interpretando resultados (no solo p‑values).

### 🤖 Semana 6 — Aprendizaje Supervisado I (Modelos Lineales y Árboles)

**🎯 Objetivos:** regresión lineal/logística, árboles de decisión, métricas.

**📖 Contenido:** train/test split, validación cruzada, RMSE, MAE, Accuracy, ROC‑AUC, PR‑AUC.

**💻 Práctica:** baseline → modelo lineal → árbol; comparación y error analysis.

**📋 Entregable:** notebook con experimento reproducible y model card breve.

### ⚙️ Semana 7 — Aprendizaje Supervisado II (Conjuntos y Feature Engineering)

**🎯 Objetivos:** feature engineering, pipelines, grid/random search.

**📖 Contenido:** One‑Hot/Target/Ordinal encoding, escalado, XGBoost/LightGBM, Optuna.

**💻 Práctica:** construir un Pipeline con preprocesamiento + modelo; tuning.

**📋 Entregable:** script train.py que guarda el mejor modelo y sus métricas.

### 🔍 Semana 8 — No Supervisado y Reducción de Dimensión

**🎯 Objetivos:** clustering (k‑means/DBSCAN), PCA/UMAP, anomaly detection.

**📖 Contenido:** silhouette score, elección de k, escalado, visualización 2D/3D.

**💻 Práctica:** segmentación de clientes y análisis de perfiles.

**📋 Entregable:** informe con segmentos y recomendaciones accionables.

### 📝 Semana 9 — Procesamiento de Texto (NLP Básico)

**🎯 Objetivos:** tokenization, tf‑idf, n‑gramas, modelos lineales para texto.

**📖 Contenido:** limpieza, stopwords, lematización; métricas para texto.

**💻 Práctica:** clasificador de sentimientos sobre reseñas.

**📋 Entregable:** notebook + matriz de confusión comentada.

### ⏰ Semana 10 — Series Temporales

**🎯 Objetivos:** descomposición, estacionariedad, lags, rolling, ARIMA básico y enfoques con ML.

**📖 Contenido:** métricas (MAE/MAPE/SMAPE), validación temporal (expanding window).

**💻 Práctica:** pronóstico de demanda semanal con variables exógenas simples.

**📋 Entregable:** notebook + gráfico de pronóstico con intervalos.

### 🗄️ Semana 11 — SQL para Analítica

**🎯 Objetivos:** SELECT/JOIN/CTE/Window Functions, agregaciones.

**📖 Contenido:** modelado de esquemas (estrella/copo), query planning básico.

**💻 Práctica:** 10–15 consultas reales (retención, cohortes, funnel).

**📋 Entregable:** archivo .sql y notebook que ejecuta queries (sqlite o PostgreSQL local).

### 🚀 Semana 12 — MLOps y Despliegue Básico

**🎯 Objetivos:** serialización (joblib), APIs con FastAPI, inference y monitoring ligero.

**📖 Contenido:** estructura de proyecto, .env, logging, pruebas unitarias, data drift.

**💻 Práctica:** empaquetar modelo y exponer POST /predict.

**📋 Entregable:** repositorio con Dockerfile (opcional), README de despliegue local.

### ⚖️ Semana 13 — Ética, Privacidad y Experimentación

**🎯 Objetivos:** sesgos, equidad, privacidad de datos, diseño de experimentos.

**📖 Contenido:** métricas de equidad (demographic parity, equalized odds), diseño de A/B test.

**💻 Práctica:** auditoría de sesgo en un dataset sintético.

**📋 Entregable:** model card ampliada con sección de riesgos y mitigaciones.

### 📱 Semana 14 — Visual Storytelling y Data Apps

**🎯 Objetivos:** comunicar para negocio.

**📖 Contenido:** narrativa de datos, dashboards en Streamlit/Plotly Dash.

**💻 Práctica:** construir una app sencilla con 2–3 vistas.

**📋 Entregable:** app ejecutable localmente + demo en video breve.

### 🎯 Semana 15 — Proyecto Integrador (Parte 1): Definición y EDA

**🎯 Objetivos:** plantear problema, definir métrica de éxito, entender datos.

**📖 Contenido:** problem framing, hipótesis, riesgos, plan de trabajo.

**💻 Práctica:** EDA profundo, data quality report y baseline.

**📋 Entregable:** propuesta de proyecto (1–2 páginas) + EDA notebook.

### 🏆 Semana 16 — Proyecto Integrador (Parte 2): Modelado, Evaluación y Entrega

**🎯 Objetivos:** experimentar, seleccionar, empaquetar, comunicar.

**📖 Contenido:** training diary, comparación honesta, post‑mortem.

**💻 Práctica:** entrenamiento final + API + dashboard.

**📋 Entregable:** repo completo con resultados, app, model card, y presentación (10 diapositivas).

## ⚡ Variantes del Plan

### 🚀 Variante Intensiva (8 semanas)

Combina cada dos semanas del plan estándar (1–2, 3–4, 5–6, 7–8, 9–10, 11–12, 13–14, 15–16).

**⏰ Dedicación:** 15–20 h/semana.

### 🐌 Variante Extendida (24 semanas)

Divide cada semana estándar en dos: primera mitad teoría/práctica guiada, segunda mitad mini‑proyecto.

## 🎯 Banco de Proyectos (Elige 1 para el Integrador)

### 📊 Opciones de Proyectos

1. **🔴 Predicción de Abandono (Churn)**
   - Clasificación binaria, foco en interpretabilidad (SHAP) y business impact.

2. **📈 Pronóstico de Demanda**
   - Serie temporal con variables externas (promociones, festivos).

3. **👥 Segmentación de Clientes**
   - Clustering + perfilado y propuestas de marketing.

4. **📝 NLP de Reseñas**
   - Análisis de sentimiento y temas; priorización de mejoras de producto.

5. **🖼️ Clasificación de Imágenes** *(Opcional DL)*
   - Defectos de fabricación con CNN pre‑entrenada.

### 📚 Fuentes de Datos Sugeridas
- **Repositorios públicos:** UCI, Kaggle, data.gov
- **Instituciones:** INE/INEGI/DANE/INE (según tu país)
- **Portales abiertos:** municipales
- **APIs públicas:** clima, transporte, etc.

## 📊 Evaluación y Rúbricas (Resumen)

### 📝 Trabajo Semanal (40%)

| Criterio | Peso |
|----------|------|
| **Correctitud técnica** | 40% |
| **Claridad del código** | 30% |
| **Comunicación de resultados** | 20% |
| **Reproducibilidad** | 10% |

### 🏆 Proyecto Integrador (60%)

| Componente | Peso |
|------------|------|
| **Planteamiento y métrica de negocio** | 15% |
| **Calidad de datos/EDA/ingeniería** | 20% |
| **Rigor experimental y métricas** | 20% |
| **Despliegue + documentación** | 25% |
| **Presentación y Q&A** | 20% |

### ✅ Definition of Done (DoD)

- **Repo completo:** README, requirements.txt o environment.yml
- **Instrucciones claras:** para reproducir
- **Datos:** incluidos o script para descargarlos
- **Notebooks limpios:** y seeds fijadas
- **Reporte final:** ≤ 8 páginas + apéndice técnico

## 🎯 Hitos y Entregables por Módulo (Lista Rápida)

| Módulos | Entregables |
|---------|-------------|
| **M1–M4** | EDA sólida y gráficos claros |
| **M5–M8** | Modelos clásicos con pipelines y validación |
| **M9–M10** | Texto y tiempo con métricas apropiadas |
| **M11–M12** | SQL + API de inferencia |
| **M13–M14** | Equidad/privacidad + app/visualización final |
| **M15–M16** | Proyecto completo con despliegue local |

## 📋 Plantillas Útiles (Copiar/Usar)

### 📁 Estructura de Repositorio

```
project/
├─ data/              # raw/processed (o scripts para descargar)
├─ notebooks/         # 01_eda.ipynb, 02_modeling.ipynb, ...
├─ src/               # etl.py, features.py, train.py, infer.py
├─ models/            # artefactos .joblib
├─ app/               # fastapi_app.py o streamlit_app.py
├─ tests/             # pruebas unitarias
├─ requirements.txt   # o environment.yml
└─ README.md
```

### ✅ Checklist de EDA (Extracto)

- [ ] **Diccionario de datos** y data quality (nulos, duplicados, outliers)
- [ ] **Variables objetivo** y leakage potencial
- [ ] **Relación variables↔objetivo** (num/cat)
- [ ] **Análisis temporal/estacionalidad** (si aplica)
- [ ] **Hipótesis de negocio** + cómo validarlas

### 📋 Card de Modelo (Resumen)

- **Propósito**
- **Población objetivo**
- **Variables principales**
- **Supuestos**
- **Riesgos y sesgos**
- **Métricas** (val/test)
- **Límites de uso**
- **Owner**

## 📚 Sugerencias de Estudio y Recursos *(No Obligatorios)*

### 📖 Libros (Referencias Clásicas)

- **Python for Data Analysis** — Wes McKinney
- **Hands‑On Machine Learning with Scikit‑Learn, Keras & TensorFlow** — Aurélien Géron
- **An Introduction to Statistical Learning (ISL, 2ª ed.)** — James et al. (con labs en R y Python)

### 💻 Práctica Guiada

- **Competencias principiantes en Kaggle:** Titanic, House Prices, SMS Spam
- **Retos de SQL:** CTEs, ventanas en plataformas de ejercicios
- **Camino opcional en R:** tidyverse (dplyr, tidyr), ggplot2, tidymodels. Transfiere conceptos 1:1

## 📅 Itinerario Sugerido por Semana (Tareas Concretas)

| Día | Actividad | Tiempo |
|-----|-----------|--------|
| **Lunes–Martes** | Teoría + notas | 2–3 h |
| **Miércoles** | Práctica guiada | 2 h |
| **Jueves** | Mini‑proyecto/experimento | 2 h |
| **Viernes** | Revisión por checklist + commit final | 1 h |
| **Fin de semana** *(opcional)* | Lectura extra/retos | 2 h |

## 🚀 Primeras 72 Horas (Para Arrancar Ya)

### ✅ Checklist de Inicio

1. **Instala Miniconda** y crea el entorno (comandos arriba)
2. **Abre JupyterLab** y completa 10 ejercicios básicos de Python (condicionales, bucles, funciones)
3. **Descarga un CSV público** (por ejemplo, ventas mensuales) y:
   - Cárgalo con pandas, inspección `.info()/.describe()`
   - Limpia nulos y crea 3 gráficos (tendencia, barras por categoría, caja)
   - Sube el repo a GitHub con un README que explique tu objetivo y hallazgos

## 🎯 Cómo Sabrás que Estás Listo/a

### ✅ Criterios de Éxito

- [ ] **Puedes plantear una pregunta de negocio**, construir un baseline, iterar y justificar por qué tu modelo final es mejor (con métricas y error analysis)
- [ ] **Tienes 3 notebooks de portafolio** (EDA sólido, modelo supervisado, serie temporal o NLP) y 1 app/endpoint de inferencia
- [ ] **Dominas consultas SQL intermedias** (joins, CTEs, ventanas)

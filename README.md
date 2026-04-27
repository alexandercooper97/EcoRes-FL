# EcoRes-FL 🌿⚡

**EcoRes-FL** es un simulador de *Federated Learning* (aprendizaje federado) consciente de la sostenibilidad y la resiliencia. Compara cuatro algoritmos bajo condiciones heterogéneas reales: clientes con distintas capacidades de cómputo, conectividad variable, fallos esporádicos y diferentes intensidades de carbono en su fuente de energía.

> *Desarrollado para investigación reproducible — genera tablas CSV y figuras PNG en resolución de publicación sin necesidad de GPUs ni conexión a internet.*

---

## Tabla de contenidos

- [¿Qué problema resuelve?](#qué-problema-resuelve)
- [Arquitectura del simulador](#arquitectura-del-simulador)
- [Algoritmos implementados](#algoritmos-implementados)
- [Instalación](#instalación)
- [Uso rápido](#uso-rápido)
- [Parámetros de línea de comandos](#parámetros-de-línea-de-comandos)
- [Datasets disponibles](#datasets-disponibles)
- [Resultados de referencia](#resultados-de-referencia)
- [Estructura de salida](#estructura-de-salida)
- [Diseño interno](#diseño-interno)
- [Ablación y sensibilidad](#ablación-y-sensibilidad)
- [Modo paper (datasets pesados)](#modo-paper-datasets-pesados)
- [Preguntas frecuentes](#preguntas-frecuentes)
- [Licencia](#licencia)

---

## ¿Qué problema resuelve?

El aprendizaje federado distribuye el entrenamiento entre muchos dispositivos (móviles, sensores IoT, servidores edge) sin centralizar los datos. Pero los enfoques clásicos como **FedAvg** ignoran tres factores críticos en el mundo real:

| Problema ignorado | Consecuencia práctica |
|---|---|
| Heterogeneidad de hardware | Clientes lentos bloquean la ronda ("stragglers") |
| Intensidad de carbono variable | Entrenar en horas pico contamina más sin razón |
| Comunicación densa | Los gradientes completos saturan la red en dispositivos IoT |

**EcoRes-FL** aborda los tres con una selección de clientes multiobjetivo + actualizaciones dispersas (top-k) + penalización proximal.

---

## Arquitectura del simulador

```
ecores_fl_enhanced.py
│
├── Carga de datos          load_federated_base_dataset()
│     └── Partición non-IID  dirichlet_partition()
│
├── Perfil de clientes      generate_client_profiles()
│     └── 4 tiers: edge-low · edge-mid · edge-gpu · cloud
│
├── Modelo local            local_train_softmax()   ← softmax + SGD + proximal term
│     └── Comunicación      sparsify_update()       ← top-k sparsification
│
├── Selección de clientes   select_clients()
│     ├── FedAvg/FedProx   → aleatorio uniforme
│     ├── CarbonAware      → score ponderado (utilidad − carbono − latencia)
│     └── EcoRes-FL        → ILP (scipy.milp) con fallback greedy
│
├── Bucle de simulación     run_method()            ← rondas federadas
│
└── Visualización           plot_lines()  plot_tradeoff()  heatmap()
```

Cada componente está desacoplado: puedes cambiar el modelo local (por ejemplo, reemplazar el softmax por una red neuronal) sin tocar la lógica de selección.

---

## Algoritmos implementados

### 1. FedAvg — *Federated Averaging* (McMahan et al., 2017)

El baseline clásico. Selecciona clientes **aleatoriamente**, agrega sus actualizaciones ponderadas por tamaño del dataset local y no aplica ninguna corrección.

- Sin penalización proximal (`μ = 0`)
- Descarta stragglers y clientes que fallan
- Gradientes completos (sin sparsification)

### 2. FedProx (Li et al., 2020)

Añade un término de regularización proximal al objetivo local:

```
F_k(w) = f_k(w) + (μ/2) · ‖w − w_global‖²
```

Esto evita que los clientes con datos muy diferentes se alejen demasiado del modelo global en cada ronda, mejorando la convergencia bajo heterogeneidad non-IID.

- Igual selección aleatoria que FedAvg
- Acepta stragglers (parcialmente)
- Sin sparsification

### 3. CarbonAware-FL

Selección determinista basada en un score multiobjetivo:

```
score = 0.60 · utilidad − 0.30 · carbono − 0.10 · latencia + 0.10 · fiabilidad
```

Prefiere clientes con baja huella de carbono instantánea y alta fiabilidad. No modifica el entrenamiento local.

### 4. EcoRes-FL ← *algoritmo propuesto*

Combina tres mejoras:

| Componente | Descripción |
|---|---|
| **Selección ILP** | Programa lineal entero binario (scipy.milp) que maximiza utilidad sujeto a presupuesto de carbono y deadline de latencia. Fallback greedy si el solver no converge. |
| **Penalización proximal** | Mismo término que FedProx (`μ = 0.02`) para estabilidad non-IID. |
| **Top-k sparsification** | Solo transmite el `sparsity`% más grande (en valor absoluto) del gradiente, reduciendo comunicación ≥ 70%. |

La intensidad de carbono de cada cliente varía sinusoidalmente por ronda (simulando energía renovable diurna) con ruido gaussiano, haciendo la selección dinámica.

---

## Instalación

### Requisitos

- Python ≥ 3.9
- Las dependencias del modo rápido son ligeras y sin GPU:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
scipy>=1.11
# Opcional para datasets de imágenes:
# torch
# torchvision
```

### Verificación rápida

```bash
python ecores_fl_enhanced.py --rounds 5 --clients 10 --skip_ablation
```

Si ves `[DONE] Results written to ./results`, todo funciona correctamente.

---

## Uso rápido

### Simulación estándar (modo rápido, ~2-5 min)

```bash
python ecores_fl_enhanced.py \
  --rounds 30 \
  --clients 40 \
  --clients_per_round 10 \
  --alpha 0.25 \
  --output_dir ./experimento_1
```

Esto ejecuta los 4 algoritmos sobre los 3 datasets por defecto (`synthetic_vision`, `iot_fault`, `digits`) y genera todas las figuras y tablas.

### Solo un dataset

```bash
python ecores_fl_enhanced.py --datasets digits --rounds 20 --skip_ablation
```

### Reproducción exacta del paper

```bash
python ecores_fl_enhanced.py \
  --rounds 30 \
  --clients 40 \
  --clients_per_round 10 \
  --alpha 0.25 \
  --seed 42
```

El seed fijo garantiza resultados idénticos en cualquier máquina con las mismas dependencias.

---

## Parámetros de línea de comandos

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `--output_dir` | str | `.` | Directorio raíz de salida |
| `--datasets` | list | `[synthetic_vision, iot_fault, digits]` | Datasets a ejecutar |
| `--rounds` | int | `30` | Número de rondas federadas |
| `--clients` | int | `40` | Total de clientes en el pool |
| `--clients_per_round` | int | `10` | Clientes seleccionados por ronda |
| `--alpha` | float | `0.25` | Concentración Dirichlet (heterogeneidad). Menor = más non-IID |
| `--seed` | int | `42` | Semilla aleatoria para reproducibilidad |
| `--paper_mode` | flag | `False` | Habilita datasets pesados (Fashion-MNIST, CIFAR-10, Covertype) |
| `--skip_ablation` | flag | `False` | Omite la grilla de ablación (más rápido) |

### Guía de parámetros

**`--alpha` (heterogeneidad)**
- `α = 0.1` → distribución de clases muy desigual entre clientes (non-IID severo)
- `α = 0.5` → heterogeneidad moderada
- `α = 1.0+` → distribución casi uniforme (casi IID)

**`--clients_per_round / --clients`**
- La fracción `clients_per_round / clients` afecta el ruido de selección. Valores típicos: 20-33%.

---

## Datasets disponibles

### Modo rápido (sin descargas)

| Dataset | Clases | Features | Muestras | Descripción |
|---|---|---|---|---|
| `synthetic_vision` | 10 | 128 | 15 000 | Clasificación multiclase con ruido (simula visión) |
| `iot_fault` | 5 | 64 | 14 000 | Detección de fallas en IoT, clases desbalanceadas |
| `digits` | 10 | 64 | ~1 800 | Dígitos manuscritos (sklearn.load_digits) |

### Modo paper (`--paper_mode`)

| Dataset | Descripción | Requiere |
|---|---|---|
| `covertype` | Clasificación de tipo de cobertura forestal | Descarga automática |
| `fashion_mnist` | Imágenes 28×28 de ropa | `torch`, `torchvision` |
| `cifar10` | Imágenes 32×32 de 10 categorías | `torch`, `torchvision` |

La partición entre clientes sigue una distribución **Dirichlet** por clase, el estándar de la literatura para simular heterogeneidad non-IID realista.

---

## Resultados de referencia

Resultados con configuración por defecto (`--rounds 20 --clients 30 --clients_per_round 8 --seed 42`):

### Dataset: `digits`

| Método | Acc. | Macro-F1 | Comm (MB) | Carbon (g) |
|---|---|---|---|---|
| FedAvg | 87.6% | 0.870 | 0.739 | 0.01545 |
| FedProx | **89.8%** | **0.896** | 0.719 | 0.01459 |
| CarbonAware | 89.1% | 0.889 | 0.754 | 0.01859 |
| **EcoRes-FL** | 88.9% | 0.887 | **0.223** | 0.01832 |

> EcoRes-FL reduce la comunicación en **~70%** respecto a FedAvg con solo 1.1% menos de precisión.

### Dataset: `iot_fault`

| Método | Acc. | Comm (MB) | Carbon (g) |
|---|---|---|---|
| FedAvg | **57.7%** | 0.364 | 0.1349 |
| FedProx | 55.9% | 0.369 | 0.1383 |
| CarbonAware | 55.3% | 0.382 | 0.1733 |
| **EcoRes-FL** | 55.8% | **0.109** | 0.1659 |

> En IoT, la reducción de comunicación es crítica: EcoRes-FL usa **3.3× menos ancho de banda** que FedAvg.

---

## Estructura de salida

Después de ejecutar el simulador, encontrarás:

```
output_dir/
├── results/
│   ├── round_history.csv          # Métricas por ronda (accuracy, carbono, comm...)
│   ├── summary_results.csv        # Resumen final por método y dataset
│   ├── ablation_synthetic_vision.csv  # Grilla de ablación (si no se omite)
│   └── experiment_metadata.json   # Configuración del experimento
│
└── figures/
    ├── convergence_{dataset}.png           # Curvas de accuracy por ronda
    ├── carbon_{dataset}.png                # Trayectoria de carbono acumulado
    ├── client_selection_heatmap_{dataset}.png  # Mapa de calor de selección EcoRes-FL
    ├── pareto_accuracy_carbon.png          # Vista Pareto accuracy-carbono
    ├── heatmap_accuracy_clients_alpha_{dataset}.png   # Ablación clientes × alpha
    ├── heatmap_carbon_clients_alpha_{dataset}.png
    ├── heatmap_accuracy_sparsity_failure_{dataset}.png  # Ablación sparsity × fallo
    └── heatmap_carbon_sparsity_failure_{dataset}.png
```

### Columnas de `round_history.csv`

| Columna | Descripción |
|---|---|
| `dataset`, `method`, `round` | Identificación |
| `accuracy`, `macro_f1` | Métricas de clasificación en test |
| `round_comm_mb`, `cum_comm_mb` | Comunicación (por ronda y acumulada) |
| `round_energy_j`, `cum_energy_j` | Energía estimada en Joules |
| `round_carbon_g`, `cum_carbon_g` | Carbono proxy en gCO₂eq |
| `selected`, `contributed`, `dropped` | Conteos de participación |
| `drop_rate` | Fracción de clientes descartados por fallo/straggler |

---

## Diseño interno

### Modelo local: softmax multiclase

El simulador usa una regresión softmax (equivalente a una red neuronal de una capa) entrenada con mini-batch SGD. Esto hace que el comportamiento sea 100% reproducible en CPU y directamente analizable.

La función de pérdida con el término proximal es:

```
L_k(w) = CrossEntropy(X_k, y_k, w) + (μ/2) · ‖w − w_global‖²
```

### Sparsificación top-k

Para reducir comunicación, EcoRes-FL solo transmite los `k` parámetros con mayor magnitud de gradiente:

```python
# keep_ratio = 0.30 → transmite el 30% más grande
flat = concat([dW.ravel(), db.ravel()])
threshold = partition(|flat|, -k)[-k]
sparse_delta = where(|delta| >= threshold, delta, 0)
```

La densidad real varía ligeramente por ronda según la distribución del gradiente.

### Intensidad de carbono dinámica

Cada cliente tiene una intensidad base `carbon_g_per_kwh` asignada según su tier. En cada ronda se modula sinusoidalmente para simular energía renovable (solar/eólica diurna):

```python
ci = base * max(0.5, 1 + 0.20 · sin(2π·r/R + φ_k) + N(0, 0.04))
```

donde `φ_k` es una fase aleatoria por cliente para romper sincronía.

### Programa lineal entero (ILP) — EcoRes-FL

Si `scipy.milp` está disponible, la selección se formula como:

```
Maximizar:  Σ_i x_i · (utility_i - λ_c·carbon_i - λ_t·latency_i + λ_r·reliability_i)
Sujeto a:   Σ_i x_i = m                    (cardinalidad)
            Σ_i x_i · latency_i ≤ m·D      (deadline promedio)
            Σ_i x_i · carbon_i ≤ budget    (presupuesto de carbono)
            x_i ∈ {0, 1}
```

Si el ILP no converge en 2 segundos, se activa el fallback greedy (ranking por score ponderado).

---

## Ablación y sensibilidad

La función `run_ablation_grid()` explora automáticamente dos grillas:

**Grilla 1: Escala de clientes × heterogeneidad**
- Clientes: 20, 40, 80
- Alpha Dirichlet: 0.10, 0.25, 0.50, 1.00

**Grilla 2: Sparsity × tasa de fallo**
- Sparsity (densidad top-k): 5%, 10%, 20%, 50%
- Failure rate: 0%, 8%, 16%, 24%

Cada combinación ejecuta 12 rondas de EcoRes-FL y genera heatmaps de accuracy y carbono.

Para ejecutar solo la ablación:

```bash
python ecores_fl_enhanced.py --rounds 12 --skip_ablation  # omite ablación estándar
# Para forzarla en un solo dataset:
# Edita args.datasets = ["synthetic_vision"] en main()
```

---

## Modo paper (datasets pesados)

Para reproducir experimentos con imágenes reales:

```bash
pip install torch torchvision

# Fashion-MNIST
python ecores_fl_enhanced.py --paper_mode --datasets fashion_mnist --rounds 30

# CIFAR-10
python ecores_fl_enhanced.py --paper_mode --datasets cifar10 --rounds 30

# Covertype (solo numpy/sklearn, descarga automática ~11 MB)
python ecores_fl_enhanced.py --paper_mode --datasets covertype --rounds 30
```

**Advertencia:** Fashion-MNIST y CIFAR-10 convierten las imágenes a vectores planos antes de la regresión softmax. Esto es intencional para mantener la comparabilidad con los datasets sintéticos, pero la accuracy absoluta será inferior a la de modelos convolucionales.

---

## Preguntas frecuentes

**¿Por qué softmax en lugar de una red neuronal profunda?**
El softmax es suficientemente expresivo para los datasets sintéticos y permite que el comportamiento sea completamente reproducible en CPU sin dependencias de GPU. Reemplazarlo por una MLP requiere solo modificar `local_train_softmax()`.

**¿Qué significa `alpha` bajo en Dirichlet?**
Con `α = 0.1`, algunos clientes tienen casi todas las muestras de una sola clase y pocas de otras. Esto simula escenarios reales donde los dispositivos de usuarios individuales tienen sesgos fuertes (un usuario solo escribe ciertos idiomas, un sensor solo detecta ciertos fallos).

**¿Por qué EcoRes-FL no siempre gana en accuracy?**
La selección de clientes por carbono y latencia puede excluir clientes con datos muy informativos si son caros energéticamente. El tradeoff intencional es: menos accuracy marginal a cambio de mucho menos comunicación y energía.

**¿Cómo conecto CodeCarbon para medición real?**
El simulador usa proxies de energía/carbono. Para medición real, instala `codecarbon` y envuelve `run_method()` con un `EmissionsTracker`. Las columnas del CSV son compatibles.

**¿El ILP siempre se activa en EcoRes-FL?**
Solo si `scipy >= 1.11` y el sistema tiene `scipy.optimize.milp` (SciPy 1.7+). Verifica con:
```python
from scipy.optimize import milp  # no debe lanzar ImportError
```
Si no está disponible, el código cae al greedy automáticamente (imprime un aviso en `experiment_metadata.json`).

---

## Citar este trabajo

Si usas este simulador en tu investigación:

```bibtex
@software{ecores_fl_2025,
  title  = {EcoRes-FL: Sustainability- and Resilience-Aware Federated Learning Simulator},
  year   = {2025},
  note   = {Anonymous for double-blind review}
}
```

---

## Licencia

Este proyecto está bajo revisión anónima. Licencia por definir tras la publicación.

---

<div align="center">
  <sub>Hecho con Python · scikit-learn · scipy · matplotlib · numpy · pandas</sub>
</div>

---

## Interfaz Web Interactiva

El proyecto incluye `index.html` — una aplicación web completa que corre **100% en el navegador**, sin servidor ni dependencias adicionales.

### ¿Por qué HTML estático y no Streamlit?

| | HTML + JS (este proyecto) | Streamlit | React |
|---|---|---|---|
| Servidor requerido | ❌ No | ✅ Sí (Python) | ✅ Sí (o build step) |
| Hosting gratuito | GitHub Pages (ilimitado) | Community Cloud (1 GB RAM, se duerme) | Vercel/Netlify |
| Latencia de simulación | 0 ms (client-side) | 2-10s (round-trip) | depende del backend |
| Usuarios simultáneos | Ilimitados | ~1-3 sin cola | depende |
| Tiempo de despliegue | < 5 minutos | ~10-15 min | > 30 min |

### Despliegue en GitHub Pages (recomendado)

```bash
# 1. Crear repositorio local
git init ecores-fl
cd ecores-fl
cp /ruta/a/index.html .

# 2. Subir a GitHub
git add index.html
git commit -m "feat: add EcoRes-FL interactive web app"
git remote add origin https://github.com/TU-USUARIO/ecores-fl.git
git push -u origin main
```

Luego en GitHub: **Settings → Pages → Source: main branch, / (root) → Save**

En ~60 segundos tu app estará disponible en:
```
https://TU-USUARIO.github.io/ecores-fl
```

### Estructura del archivo web

```
index.html  (archivo único, ~25 KB)
│
├── CSS       Tema oscuro futurista, diseño responsivo
├── HTML      Controles, gráficas, tabla, sección de despliegue
├── Chart.js  Cargado desde CDN (cdnjs.cloudflare.com)
└── JS        Datos pre-computados embebidos + lógica de visualización
```

Los datos de simulación están **pre-computados** y embebidos directamente en el HTML como JSON, por lo que no se necesita conexión a una API en tiempo de ejecución.

### Funcionalidades de la interfaz

- Selector de dataset (`synthetic_vision`, `iot_fault`, `digits`)
- Slider de rondas (5 a 15)
- Slider de alpha Dirichlet (0.10 a 1.00)
- Toggle por método (activar/desactivar cada algoritmo)
- 4 gráficas interactivas: accuracy, comunicación, carbono, Pareto
- Tarjetas de métricas con comparativa EcoRes-FL vs FedAvg
- Tabla comparativa completa con deltas de comunicación

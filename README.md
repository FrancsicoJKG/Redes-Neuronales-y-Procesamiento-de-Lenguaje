# UMG — AI Sentiment Analyzer

Laboratorio interactivo de análisis de sentimiento construido con **TensorFlow.js** y **NLP** en el navegador. No requiere servidor ni backend — todo el entrenamiento e inferencia ocurre directamente en el cliente.


---

## ¿Qué hace?

Permite entrenar una red neuronal pequeña con frases de ejemplo (positivas y negativas), y luego clasificar nuevas frases en tiempo real mostrando:

- Etiqueta **POSITIVO** o **NEGATIVO**
- Porcentaje de confianza del modelo
- Barra de confianza animada
- Tokens (palabras) coloreados según su polaridad
- Gráficas de pérdida y precisión por época

---

## Tecnologías

| Tecnología | Uso |
|---|---|
| TensorFlow.js v4.15 | Construcción y entrenamiento de la red neuronal |
| HTML / CSS / JS puro | Interfaz sin frameworks externos |
| Canvas API | Gráficas de pérdida y precisión |

---

## Estructura del proyecto

```
proyecto/
│
├── README.md       → Documentación general del proyecto
├── index.html      → Contiene la estructura principal de la página
├── style.css       → Contiene los estilos visuales de la aplicación
└── script.js       → Contiene la lógica e interactividad del sistema
```
---

## Cómo usarlo

1. Abre el archivo `index.html` en cualquier navegador moderno.
2. Ajusta los hiperparámetros con los sliders (opcional).
3. Haz clic en **▶ Entrenar modelo**.
4. Escribe una frase técnica en el área de texto.
5. Haz clic en **Ejecutar sentimiento** para ver la predicción.

---

## Arquitectura de la red neuronal

```
Entrada (22 tokens)
      ↓
Dense(hidden, relu)     ← neuronas configurables via slider
      ↓
Dropout(0.18)           ← regularización para evitar overfitting
      ↓
Dense(hidden/2, relu)
      ↓
Dense(1, sigmoid)       ← salida: probabilidad 0.0 a 1.0
```

- **Optimizador:** Adam
- **Función de pérdida:** Binary Crossentropy
- **Secuencia máxima:** 22 tokens por frase

---

## Hiperparámetros ajustables

| Parámetro | Rango | Por defecto |
|---|---|---|
| Tasa de aprendizaje | 0.001 – 0.080 | 0.010 |
| Épocas | 30 – 420 | 160 |
| Unidades ocultas | 8 – 96 | 32 |

---

## Dataset inicial

El modelo viene precargado con **20 frases** de contexto técnico/DevOps:

- **10 positivas:** frases sobre sistemas estables, despliegues exitosos, buen rendimiento.
- **10 negativas:** frases sobre errores, caídas, latencia alta, fallos de configuración.

Puedes agregar tus propias frases desde la interfaz usando los botones **+ POS** y **+ NEG**.

---

## Tokenizador

Se implementa un tokenizador simple desde cero (`SimpleTokenizer`) que:

1. Convierte el texto a minúsculas y elimina tildes y caracteres especiales.
2. Construye un vocabulario ordenado por frecuencia a partir del dataset.
3. Codifica cada frase como una secuencia numérica de longitud fija (padding con ceros).

Los tokens especiales reservados son `<PAD>` (relleno) y `<UNK>` (palabra desconocida).

---

## Interpretación de resultados

| Resultado | Significado |
|---|---|
| Confianza > 80% | Clasificación clara según el dataset |
| Confianza ~50% | Incertidumbre: frase ambigua, sarcasmo o vocabulario nuevo |
| Tokens en verde | Palabras asociadas a frases positivas |
| Tokens en rojo | Palabras asociadas a frases negativas |
| Tokens en gris | Palabras neutras o no reconocidas |

---

## Limitaciones

- El modelo es demostrativo y se entrena con muy pocas muestras (20 por defecto).
- No persiste entre recargas de página (el entrenamiento debe repetirse).
- El tokenizador no maneja sinónimos ni contexto semántico profundo.
- Funciona mejor con frases en el mismo dominio del dataset de entrenamiento.

---

## Compatibilidad

Funciona en cualquier navegador moderno con soporte para:
- `WebGL` o `WASM` (requerido por TensorFlow.js)
- `Canvas API`
- `ES6+`

Probado en Chrome, Firefox y Edge.

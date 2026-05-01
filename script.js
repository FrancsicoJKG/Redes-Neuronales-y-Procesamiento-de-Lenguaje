/* ══════════════════════════════════════════════
   Constantes y estado global
   ══════════════════════════════════════════════ */

/* Longitud máxima de la secuencia de tokens por frase */
const MAX_LEN = 22;

/* Dataset inicial: 10 frases positivas (label:1) y 10 negativas (label:0) */
let dataset = [
  { text: "El servidor Proxmox es muy estable y confiable",                  label: 1 },
  { text: "Docker simplifica la configuración del entorno",                  label: 1 },
  { text: "La API REST responde rápido y sin errores",                       label: 1 },
  { text: "El despliegue automatizado funcionó correctamente",               label: 1 },
  { text: "El sistema de monitoreo detectó el problema a tiempo",            label: 1 },
  { text: "La base de datos PostgreSQL mantiene buen rendimiento",           label: 1 },
  { text: "El modelo de inteligencia artificial clasificó correctamente",    label: 1 },
  { text: "La autenticación con JWT funciona de manera segura",              label: 1 },
  { text: "El contenedor se ejecuta estable en producción",                  label: 1 },
  { text: "La interfaz es clara y permite analizar datos fácilmente",        label: 1 },

  { text: "El despliegue en Docker falló por falta de memoria",              label: 0 },
  { text: "El servidor se cayó por un error crítico",                        label: 0 },
  { text: "La aplicación se congela constantemente",                         label: 0 },
  { text: "La base de datos perdió registros importantes",                   label: 0 },
  { text: "El pipeline de Jenkins se rompe en producción",                   label: 0 },
  { text: "La latencia de red es terrible e inaceptable",                    label: 0 },
  { text: "El modelo genera predicciones incorrectas",                       label: 0 },
  { text: "La autenticación falla con tokens expirados",                     label: 0 },
  { text: "El sistema tiene errores de configuración",                       label: 0 },
  { text: "El servicio responde lento y causa interrupciones",               label: 0 },
];

/* Variables globales del modelo */
let tokenizer = null;
let model     = null;
let losses    = [];
let accuracies = [];

/* ══════════════════════════════════════════════
   Tokenizador simple
   Convierte texto a secuencias numéricas
   ══════════════════════════════════════════════ */
class SimpleTokenizer {
  constructor() {
    /* Índice de palabras: 0 = padding, 1 = desconocida */
    this.word2idx = { "<PAD>": 0, "<UNK>": 1 };
    this.idx2word = ["<PAD>", "<UNK>"];
  }

  /* Limpia el texto: minúsculas, sin tildes, sin caracteres especiales */
  clean(text) {
    return text
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter(w => w.length > 1);
  }

  /* Construye el vocabulario a partir de todas las frases del dataset */
  fit(texts) {
    const freq = {};
    texts.forEach(t => this.clean(t).forEach(w => freq[w] = (freq[w] || 0) + 1));
    Object.entries(freq)
      .sort((a, b) => b[1] - a[1])
      .forEach(([w]) => {
        if (this.word2idx[w] === undefined) {
          this.word2idx[w] = this.idx2word.length;
          this.idx2word.push(w);
        }
      });
  }

  /* Convierte una frase a un array numérico de longitud MAX_LEN (con padding) */
  encode(text) {
    const ids = this.clean(text).map(w => this.word2idx[w] ?? 1);
    while (ids.length < MAX_LEN) ids.push(0);
    return ids.slice(0, MAX_LEN);
  }
}

/* ══════════════════════════════════════════════
   Funciones de UI
   ══════════════════════════════════════════════ */

/* Renderiza la lista de frases del dataset en el HTML */
function renderDataset() {
  const list = document.getElementById("dataset-list");
  list.innerHTML = dataset.map(d => `
    <div class="sample">
      <span class="tag ${d.label ? "pos" : "neg"}">${d.label ? "POS" : "NEG"}</span>
      <p>${d.text}</p>
    </div>
  `).join("");
  document.getElementById("side-samples").textContent = dataset.length;
}

/* Agrega una nueva frase al dataset según el botón presionado (POS o NEG) */
function addSample(label) {
  const input = document.getElementById("new-sample");
  const value = input.value.trim();
  if (!value) return;
  dataset.push({ text: value, label });
  input.value = "";
  renderDataset();
}

/* Sincroniza los valores de los sliders con sus etiquetas numéricas */
function syncParams() {
  document.getElementById("lr-label").textContent = (document.getElementById("lr").value / 1000).toFixed(3);
  document.getElementById("ep-label").textContent = document.getElementById("epochs").value;
  document.getElementById("hu-label").textContent = document.getElementById("hidden").value;
}

/* Agrega una línea al log con un color opcional (ok / warn / bad) */
function logLine(text, cls = "") {
  const log = document.getElementById("log");
  const div = document.createElement("div");
  div.className  = cls;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

/* Actualiza el estado del modelo en la UI (métricas + sidebar) */
function setState(text) {
  document.getElementById("m-state").textContent    = text;
  document.getElementById("side-model").textContent = text;
}

/* ══════════════════════════════════════════════
   Gráficas con Canvas API
   ══════════════════════════════════════════════ */

/* Dibuja una gráfica de línea en el canvas indicado */
function drawChart(id, values, color) {
  const canvas = document.getElementById(id);
  const parent = canvas.parentElement;
  const ctx    = canvas.getContext("2d");
  const W      = parent.clientWidth - 24;
  const H      = 120;

  canvas.width  = W;
  canvas.height = H;
  ctx.clearRect(0, 0, W, H);

  /* Líneas de referencia horizontales */
  ctx.strokeStyle = "rgba(255,255,255,.06)";
  ctx.lineWidth   = 1;
  for (let i = 1; i < 4; i++) {
    ctx.beginPath();
    ctx.moveTo(0, H * i / 4);
    ctx.lineTo(W, H * i / 4);
    ctx.stroke();
  }

  if (values.length < 2) return;

  const max   = Math.max(...values, 1);
  const min   = Math.min(...values, 0);
  const range = max - min || 1;
  const step  = W / (values.length - 1);

  /* Dibuja la curva de valores */
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = i * step;
    const y = H - ((v - min) / range) * (H - 14) - 7;
    if (i === 0) ctx.moveTo(x, y);
    else         ctx.lineTo(x, y);
  });
  ctx.stroke();
}

/* ══════════════════════════════════════════════
   Entrenamiento de la red neuronal
   ══════════════════════════════════════════════ */
async function train() {
  const trainBtn   = document.getElementById("train-btn");
  const analyzeBtn = document.getElementById("analyze-btn");

  trainBtn.disabled   = true;
  analyzeBtn.disabled = true;
  setState("ENTRENANDO");
  document.getElementById("train-state").textContent = "MODELO::ENTRENANDO";
  document.getElementById("log").innerHTML = "";
  losses     = [];
  accuracies = [];

  /* Lee los hiperparámetros de los sliders */
  const lr     = Number(document.getElementById("lr").value) / 1000;
  const epochs = Number(document.getElementById("epochs").value);
  const hidden = Number(document.getElementById("hidden").value);

  /* Tokeniza el dataset completo */
  tokenizer = new SimpleTokenizer();
  tokenizer.fit(dataset.map(d => d.text));

  document.getElementById("m-vocab").textContent    = tokenizer.idx2word.length;
  document.getElementById("side-vocab").textContent = tokenizer.idx2word.length;

  /* Convierte frases a tensores de entrada (xs) y etiquetas (ys) */
  const xs = tf.tensor2d(
    dataset.map(d => tokenizer.encode(d.text)),
    [dataset.length, MAX_LEN],
    "float32"
  );
  const ys = tf.tensor2d(
    dataset.map(d => [d.label]),
    [dataset.length, 1],
    "float32"
  );

  if (model) model.dispose();

  /* Arquitectura: Dense(relu) → Dropout → Dense(relu) → Dense(sigmoid) */
  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [MAX_LEN], units: hidden, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: .18 }));
  model.add(tf.layers.dense({ units: Math.max(4, Math.round(hidden / 2)), activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(lr),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  logLine(`> tokenizador ajustado | vocab=${tokenizer.idx2word.length}`, "ok");
  logLine(`> entrenamiento iniciado | lr=${lr.toFixed(3)} | épocas=${epochs} | ocultas=${hidden}`);

  let finalAcc  = 0;
  let finalLoss = 0;

  /* Entrena el modelo y actualiza el log y las gráficas cada cierto número de épocas */
  await model.fit(xs, ys, {
    epochs,
    batchSize: Math.min(8, dataset.length),
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        finalLoss = logs.loss;
        finalAcc  = logs.acc ?? logs.accuracy ?? 0;
        losses.push(finalLoss);
        accuracies.push(finalAcc);

        if (epoch === 0 || (epoch + 1) % 10 === 0 || epoch + 1 === epochs) {
          const cls = finalAcc > .92 ? "ok" : finalAcc < .65 ? "warn" : "";
          logLine(
            `época ${String(epoch + 1).padStart(3, "0")}/${epochs} | pérdida=${finalLoss.toFixed(4)} | prec=${(finalAcc * 100).toFixed(1)}%`,
            cls
          );
        }

        if ((epoch + 1) % 5 === 0) {
          drawChart("lossCanvas", losses,     "#ff4b6e");
          drawChart("accCanvas",  accuracies, "#44d8ff");
        }
      },
    },
  });

  /* Libera memoria de los tensores de entrenamiento */
  xs.dispose();
  ys.dispose();

  /* Actualiza gráficas y métricas finales */
  drawChart("lossCanvas", losses,     "#ff4b6e");
  drawChart("accCanvas",  accuracies, "#44d8ff");

  document.getElementById("m-acc").textContent       = `${(finalAcc * 100).toFixed(1)}%`;
  document.getElementById("m-loss").textContent      = finalLoss.toFixed(3);
  document.getElementById("train-state").textContent = "MODELO::LISTO";
  setState("LISTO");
  logLine("> modelo listo para inferencia", "ok");

  trainBtn.disabled   = false;
  analyzeBtn.disabled = false;
}

/* ══════════════════════════════════════════════
   Inferencia: clasificar una frase nueva
   ══════════════════════════════════════════════ */
async function analyze() {
  if (!model || !tokenizer) {
    alert("Primero entrena el modelo.");
    return;
  }

  const input = document.getElementById("text").value.trim();
  if (!input) return;

  /* Codifica la frase y obtiene la predicción del modelo */
  const encoded = tokenizer.encode(input);
  const x       = tf.tensor2d([encoded], [1, MAX_LEN], "float32");
  const pred    = await model.predict(x).data();
  x.dispose();

  /* pred[0] es la probabilidad de ser positivo (0 a 1) */
  const pos   = pred[0];
  const neg   = 1 - pos;
  const isPos = pos >= .5;
  const conf  = Math.max(pos, neg);

  /* Muestra la etiqueta POSITIVO o NEGATIVO con su color */
  const label = document.getElementById("pred-label");
  label.className  = isPos ? "pos" : "neg";
  label.textContent = isPos ? "▲ POSITIVO" : "▼ NEGATIVO";

  document.getElementById("pred-info").textContent =
    `Confianza del modelo: ${(conf * 100).toFixed(1)}%. ` +
    (Math.abs(pos - .5) < .12
      ? "Existe incertidumbre porque el resultado está cerca del 50%."
      : "La clasificación es relativamente clara según el dataset.");

  /* Actualiza la barra de confianza y los porcentajes */
  document.getElementById("conf-bar").style.width = `${(conf * 100).toFixed(1)}%`;
  document.getElementById("pct-line").textContent =
    `Positivo: ${(pos * 100).toFixed(1)}% | Negativo: ${(neg * 100).toFixed(1)}%`;

  /* Listas de palabras clave para colorear los tokens */
  const positiveWords = ["estable","rapido","correctamente","segura","seguro","funciona","rendimiento","clara","facilmente","confiable","automatizado"];
  const negativeWords = ["fallo","error","critico","congela","perdio","rompe","terrible","inaceptable","incorrectas","falla","lento","interrupciones"];

  /* Renderiza cada token con su color según su polaridad */
  const tokens = tokenizer.clean(input);
  document.getElementById("tokens").innerHTML = tokens.map(t => {
    const cls = positiveWords.some(w => t.includes(w)) ? "good"
              : negativeWords.some(w => t.includes(w)) ? "bad"
              : "";
    return `<span class="token ${cls}">${t}</span>`;
  }).join("");
}

/* ══════════════════════════════════════════════
   Utilidades de interfaz
   ══════════════════════════════════════════════ */

/* Carga una frase de ejemplo aleatoria en el textarea */
function loadDemo() {
  const examples = [
    "El servidor Proxmox es estable y responde rápido",
    "El despliegue en Docker falló por falta de memoria",
    "La API REST funciona correctamente con autenticación JWT",
    "El servicio responde lento y causa interrupciones",
    "La interfaz se ve bien pero el modelo genera resultados confusos",
  ];
  document.getElementById("text").value =
    examples[Math.floor(Math.random() * examples.length)];
}

/* Limpia el área de resultados y el textarea */
function resetOutput() {
  document.getElementById("text").value            = "";
  document.getElementById("pred-label").className  = "";
  document.getElementById("pred-label").textContent = "En espera";
  document.getElementById("pred-info").textContent  = "Entrena el modelo y luego escribe una frase para clasificarla.";
  document.getElementById("conf-bar").style.width   = "0%";
  document.getElementById("pct-line").textContent   = "Positivo: — | Negativo: —";
  document.getElementById("tokens").innerHTML       = "";
}

/* ══════════════════════════════════════════════
   Inicialización al cargar la página
   ══════════════════════════════════════════════ */
renderDataset();
syncParams();

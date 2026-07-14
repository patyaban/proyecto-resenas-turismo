// Lógica de la interfaz web del analizador de reseñas de turismo.
// Se comunica con la API FastAPI mediante fetch (mismo origen).

const $ = (sel) => document.querySelector(sel);

const CONFIG_SENTIMIENTO = {
  positivo: { emoji: "😊" },
  negativo: { emoji: "😠" },
  neutral:  { emoji: "😐" },
};

const CONFIG_ASPECTO = {
  positivo: "pos",
  negativo: "neg",
  neutral: "neu",
  no_mencionado: "na",
};

const ICONO_ASPECTO = {
  comida: "🍽️",
  servicio: "🛎️",
  limpieza: "🧼",
  precio: "💰",
};

const EJEMPLOS = [
  "La comida estuvo deliciosa y el servicio fue muy atento, aunque los precios son un poco altos.",
  "Pésima limpieza en la habitación y el personal fue descortés, no lo recomiendo.",
  "El desayuno es aceptable, cumple, pero nada memorable.",
  "Excelente relación precio-calidad, el lugar estaba impecable y nos atendieron muy bien.",
  "El servicio fue muy lento y la comida llegó fría, una decepción total.",
];

// --- Análisis de una reseña --------------------------------------------------

async function analizar() {
  const texto = $("#entrada").value.trim();
  if (!texto) {
    mostrarError("Pega una reseña primero.");
    return;
  }
  const modelo = document.querySelector('input[name="modelo"]:checked').value;

  ocultar("#resultado");
  ocultar("#error");
  mostrar("#cargando");

  try {
    const resp = await fetch("/analizar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto, modelo }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `Error ${resp.status}`);
    }
    const data = await resp.json();
    mostrarResultado(data);
  } catch (e) {
    mostrarError(e.message);
  } finally {
    ocultar("#cargando");
  }
}

function mostrarResultado(data) {
  // Sentimiento general
  const conf = CONFIG_SENTIMIENTO[data.sentimiento_general] || CONFIG_SENTIMIENTO.neutral;
  $("#emoji-resultado").textContent = conf.emoji;
  $("#sentimiento-texto").textContent = data.sentimiento_general;
  $("#confianza-texto").textContent =
    `Confianza: ${(data.confianza * 100).toFixed(1)}% · modelo ${data.modelo_usado}`;

  // Estrellas (con animación escalonada, ver estilos.css @keyframes llenar)
  const contEstrellas = $("#estrellas");
  contEstrellas.innerHTML = "";
  for (let i = 0; i < 5; i++) {
    const s = document.createElement("span");
    s.textContent = i < data.estrellas_estimadas ? "★" : "☆";
    s.style.animationDelay = `${i * 0.06}s`;
    contEstrellas.appendChild(s);
  }

  // Aspectos
  const contAspectos = $("#aspectos");
  contAspectos.innerHTML = "";
  for (const [aspecto, valor] of Object.entries(data.aspectos)) {
    const tarjeta = document.createElement("div");
    tarjeta.className = "aspecto-tarjeta";
    const claseValor = CONFIG_ASPECTO[valor] || "na";
    const etiquetaValor = valor === "no_mencionado" ? "no mencionado" : valor;
    tarjeta.innerHTML = `
      <div class="aspecto-icono">${ICONO_ASPECTO[aspecto] || "•"}</div>
      <div class="aspecto-nombre">${aspecto}</div>
      <div class="aspecto-valor ${claseValor}">${etiquetaValor}</div>`;
    contAspectos.appendChild(tarjeta);
  }

  // Tema dominante (si el modelo LDA fue entrenado)
  if (data.tema_dominante) {
    $("#tema-palabras").textContent = data.tema_dominante.palabras_clave.slice(0, 6).join(", ");
    mostrar("#tema-dominante");
  } else {
    ocultar("#tema-dominante");
  }

  mostrar("#resultado");
}

// --- Panel lateral: comparación de modelos -----------------------------------

async function cargarMetricas() {
  try {
    const resp = await fetch("/metricas");
    if (!resp.ok) throw new Error();
    const m = await resp.json();
    const clasico = m.clasico || {};
    const neuronal = m.neuronal || {};
    const mejor = m.mejor_modelo_por_f1_macro;

    const fila = (etiqueta, clave) => {
      const vc = clasico[clave] !== undefined ? (clasico[clave] * 100).toFixed(1) + "%" : "—";
      const vn = neuronal[clave] !== undefined ? (neuronal[clave] * 100).toFixed(1) + "%" : "—";
      const claseC = mejor === "clasico" && clave === "f1_macro" ? "mejor" : "";
      const claseN = mejor === "neuronal" && clave === "f1_macro" ? "mejor" : "";
      return `<tr><td>${etiqueta}</td><td class="${claseC}">${vc}</td><td class="${claseN}">${vn}</td></tr>`;
    };

    const tablaMatriz = (matriz, clases) => {
      if (!matriz || !clases) return "<p class='sin-datos'>Sin matriz disponible.</p>";
      const encabezado = clases.map(c => `<th>${c.slice(0, 3)}</th>`).join("");
      const filas = matriz.map((fila, i) => `
        <tr><td>${clases[i].slice(0, 3)}</td>${fila.map(v => `<td>${v}</td>`).join("")}</tr>
      `).join("");
      return `<table class="matriz"><thead><tr><th>real\\pred</th>${encabezado}</tr></thead><tbody>${filas}</tbody></table>`;
    };

    $("#comparacion").innerHTML = `
      <table>
        <thead><tr><th></th><th>Clásico</th><th>Neuronal</th></tr></thead>
        <tbody>
          ${fila("Exactitud", "exactitud")}
          ${fila("Precisión macro", "precision_macro")}
          ${fila("Recall macro", "recall_macro")}
          ${fila("F1 macro", "f1_macro")}
        </tbody>
      </table>
      <p class="subtitulo-matriz">Matriz de confusión — Clásico</p>
      ${tablaMatriz(clasico.matriz_confusion, clasico.clases)}
      <p class="subtitulo-matriz">Matriz de confusión — Neuronal</p>
      ${tablaMatriz(neuronal.matriz_confusion, neuronal.clases)}`;
  } catch {
    $("#comparacion").textContent = "No disponibles (entrena ambos modelos y corre comparar_modelos.py).";
  }
}

// --- Panel lateral: temas LDA -------------------------------------------------

async function cargarTemas() {
  try {
    const resp = await fetch("/temas");
    if (!resp.ok) throw new Error();
    const data = await resp.json();
    $("#temas").innerHTML = data.temas.map(t => `
      <div class="tema-fila"><b>Tema ${t.tema}:</b> ${t.palabras_clave.slice(0, 8).join(", ")}</div>
    `).join("");
  } catch {
    $("#temas").textContent = "No disponibles (ejecuta python -m src.modelado_temas primero).";
  }
}

// --- Utilidades de UI ---------------------------------------------------------

function mostrar(sel) { $(sel).classList.remove("oculto"); }
function ocultar(sel) { $(sel).classList.add("oculto"); }
function mostrarError(msg) {
  const e = $("#error");
  e.textContent = "⚠️ " + msg;
  mostrar("#error");
}

function crearEjemplos() {
  const cont = $("#ejemplos");
  EJEMPLOS.forEach((txt) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.textContent = txt.length > 46 ? txt.slice(0, 46) + "…" : txt;
    chip.title = txt;
    chip.addEventListener("click", () => {
      $("#entrada").value = txt;
      analizar();
    });
    cont.appendChild(chip);
  });
}

// --- Inicialización ------------------------------------------------------------

$("#btn-analizar").addEventListener("click", analizar);
$("#btn-limpiar").addEventListener("click", () => {
  $("#entrada").value = "";
  ocultar("#resultado");
  ocultar("#error");
});
$("#entrada").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analizar();
});

crearEjemplos();
cargarMetricas();
cargarTemas();

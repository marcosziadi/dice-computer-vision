# -*- coding: utf-8 -*-
"""
TP3 - Procesamiento de Imágenes - Problema: Cinco dados
=======================================================
Para cada video tirada_<id>.mp4 de la carpeta:

  A) Detecta los frames en los que los dados están quietos (mirando solo los píxeles rojos).
  B) Informa por consola el valor de cada dado.
  C) Genera resultado_tirada_<id>.mp4 con la bounding box, el id y el valor de cada dado
     desde que están en reposo (ej.: D1-5, D2-2, D3-2, D4-2, D5-3). La caja de cada dado se
     mantiene hasta el frame en el que ya no queda ningún píxel rojo dentro de ella.

Uso:
    python tp3_cinco_dados.py              # procesa los tirada_*.mp4 de la carpeta del script
    python tp3_cinco_dados.py <carpeta>    # o los de otra carpeta

Lógica para leer los dados (en el frame con los dados quietos):
    HSV -> píxeles rojos -> clausura (rellena los puntos) -> apertura (saca ruido)
    -> componentes conexas -> se queda con las que son cuadradas (área + forma)
    -> Otsu usando solo los píxeles de cada dado -> apertura sobre la binarización
    -> componentes conexas -> se queda con las redondas (área + ejes por momentos)
    -> cantidad de puntos = valor del dado.
"""
import glob
import os
import re
import sys

import cv2
import numpy as np

# =====================================================================================
# PARÁMETROS
# =====================================================================================
# --- Máscara roja (HSV) ---
S_MIN, V_MIN = 150, 60
ESCALA = 1 / 3                 # la detección de reposo usa frames reducidos (más rápido)

# --- A) Frames con los dados quietos ---
AREA_ROJA_MIN = 1000           # área roja mínima para considerar que hay dados
VENTANA = 2                    # el frame de referencia se elige promediando el cambio en +-2 frames
TOL = 5                        # tolerancia (px) al comparar contra el frame quieto
UMBRAL = 0.05                  # diferencia máxima con el frame quieto (fracción de su área roja)

# --- Dados ---
TAM_CIERRE = 15                # clausura: rellena los puntos de los dados
TAM_APERTURA = 11              # apertura (erosión + dilatación) sobre la máscara de dados
AREA_MIN_DADO, AREA_MAX_DADO = 1500, 12000
CUADRADO_MIN = 0.70            # área / lado_mayor^2 del rectángulo base (cuadrado perfecto = 1; dados reales 0.78-0.93)
TAM_BORDE = 5                  # se erosiona el dado para no analizar su borde
PAD = 10                       # margen al recortar cada dado

# --- Puntos ---
TAM_PUNTOS = 3                 # apertura sobre la binarización de Otsu
AREA_MIN_PUNTO, AREA_MAX_PUNTO = 20, 400
EJES_MIN_PUNTO = 0.80          # eje menor / eje mayor (círculo = 1)

# --- Dibujo ---
COLOR_CAJA = (0, 255, 255)     # amarillo (BGR)
ROJO_MIN_EN_CAJA = 1           # la caja se mantiene mientras haya al menos estos píxeles rojos dentro

KERNEL3 = np.ones((3, 3), np.uint8)
KERNEL_TOL = np.ones((TOL, TOL), np.uint8)
KERNEL_CIERRE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TAM_CIERRE, TAM_CIERRE))
KERNEL_APERTURA = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TAM_APERTURA, TAM_APERTURA))
KERNEL_BORDE = np.ones((TAM_BORDE, TAM_BORDE), np.uint8)
KERNEL_PUNTOS = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TAM_PUNTOS, TAM_PUNTOS))


# =====================================================================================
# A) FRAMES CON LOS DADOS QUIETOS
# =====================================================================================
def mascara_roja(frame_bgr):
    """Pasa a HSV y deja solo los píxeles rojos (el rojo está en los dos extremos de H)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m = (cv2.inRange(hsv, (0, S_MIN, V_MIN), (10, 255, 255)) |
         cv2.inRange(hsv, (170, S_MIN, V_MIN), (180, 255, 255)))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL3)      # saca líneas finas y ruido


def leer_mascaras(path):
    """Máscara roja (a escala reducida) de todos los frames del video."""
    cap = cv2.VideoCapture(path)
    mascaras = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, None, fx=ESCALA, fy=ESCALA, interpolation=cv2.INTER_AREA)
        mascaras.append(mascara_roja(frame))
    cap.release()
    return mascaras


def leer_frame(path, n_frame):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def diferencia_tolerante(m, ref):
    """Píxeles rojos de m lejos de los de ref + píxeles rojos de ref sin rojo cerca en m."""
    nuevos = cv2.countNonZero(cv2.bitwise_and(m, cv2.bitwise_not(cv2.dilate(ref, KERNEL_TOL))))
    perdidos = cv2.countNonZero(cv2.bitwise_and(ref, cv2.bitwise_not(cv2.dilate(m, KERNEL_TOL))))
    return nuevos + perdidos


def rango_quieto(mascaras):
    """Devuelve (ref, ini, fin, dif): frame de referencia y rango de frames con los dados quietos.
    Devuelve None si en ningún frame se ven dados."""
    n = len(mascaras)
    areas = np.array([cv2.countNonZero(m) for m in mascaras])
    cambios = np.array([0] + [cv2.countNonZero(cv2.bitwise_xor(mascaras[k], mascaras[k - 1]))
                              for k in range(1, n)])

    # 1) Referencia: menor cambio promedio en una ventana, con dados visibles en toda ella
    prom = np.full(n, np.inf)
    for t in range(VENTANA + 1, n - VENTANA):
        if areas[t - VENTANA:t + VENTANA + 1].min() >= AREA_ROJA_MIN:
            prom[t] = cambios[t - VENTANA:t + VENTANA + 1].mean()
    if not np.isfinite(prom).any():
        return None
    ref = int(np.argmin(prom))

    # 2) Diferencia de cada frame contra el de referencia (solo píxeles rojos)
    dif = np.array([diferencia_tolerante(m, mascaras[ref]) for m in mascaras]) / areas[ref]

    # 3) Se extiende hacia atrás y hacia adelante mientras la diferencia sea <= UMBRAL
    ini = ref
    while ini > 0 and dif[ini - 1] <= UMBRAL:
        ini -= 1
    fin = ref
    while fin < n - 1 and dif[fin + 1] <= UMBRAL:
        fin += 1
    return ref, ini, fin, dif


# =====================================================================================
# DESCRIPTORES DE FORMA
# =====================================================================================
def indice_cuadrado(comp):
    """área / lado_mayor^2 del rectángulo base (el menor rectángulo rotado que contiene la región).
    Es rectangularidad / alargamiento: vale 1 en un cuadrado y baja si la figura se alarga
    o no llena su rectángulo (por ejemplo, dos dados pegados)."""
    pts = cv2.findNonZero(comp)
    if pts is None:
        return 0.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    lado_mayor = max(rw, rh)
    return cv2.countNonZero(comp) / lado_mayor ** 2 if lado_mayor > 0 else 0.0


def relacion_ejes(comp):
    """Eje menor / eje mayor de una región binaria, calculados con momentos centrales.
    Vale 1 en un círculo y baja a medida que la figura se alarga."""
    M = cv2.moments(comp, binaryImage=True)
    if M["m00"] == 0:
        return 0.0
    mu20 = M["mu20"] / M["m00"]            # varianza en x
    mu02 = M["mu02"] / M["m00"]            # varianza en y
    mu11 = M["mu11"] / M["m00"]            # covarianza x-y
    d = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    l1 = (mu20 + mu02) / 2 + d             # dispersión a lo largo del eje mayor
    l2 = (mu20 + mu02) / 2 - d             # dispersión a lo largo del eje menor
    if l1 <= 0:
        return 0.0
    return float(np.sqrt(max(l2, 0.0) / l1))


# =====================================================================================
# B) LECTURA DE LOS DADOS
# =====================================================================================
def clasificar_puntos(binaria):
    """Componentes conexas de la binarización: devuelve (cantidad de puntos, lista de blobs, etiquetas).
    Un blob es un punto si su área está en rango y sus ejes son parecidos (redondo)."""
    n, et, st, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    blobs = []
    for k in range(1, n):
        area = int(st[k, cv2.CC_STAT_AREA])
        if AREA_MIN_PUNTO <= area <= AREA_MAX_PUNTO:
            ejes = relacion_ejes((et == k).astype(np.uint8) * 255)
            ok = ejes >= EJES_MIN_PUNTO
        else:
            ejes, ok = None, False
        blobs.append({"k": k, "area": area, "ejes": ejes, "ok": ok})
    return sum(b["ok"] for b in blobs), blobs, et


def detectar_dados(frame, pasos=None):
    """Detecta y lee los dados de un frame con los dados quietos.
    Devuelve una lista de dicts con: id, bbox (x, y, w, h), centro y valor.
    Si se pasa un dict en `pasos`, guarda ahí las imágenes intermedias (para graficar)."""
    H, W = frame.shape[:2]

    # 1) HSV -> píxeles rojos
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rojo = (cv2.inRange(hsv, (0, S_MIN, V_MIN), (10, 255, 255)) |
            cv2.inRange(hsv, (170, S_MIN, V_MIN), (180, 255, 255)))

    # 2) Clausura (rellena los puntos) y apertura = erosión + dilatación (saca ruido)
    cerrada = cv2.morphologyEx(rojo, cv2.MORPH_CLOSE, KERNEL_CIERRE)
    abierta = cv2.morphologyEx(cerrada, cv2.MORPH_OPEN, KERNEL_APERTURA)

    # 3) Componentes conexas: cada dado es una componente
    n, et, stats, _ = cv2.connectedComponentsWithStats(abierta, connectivity=8)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dados, clase, indices = [], {}, {}
    for k in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[k]]
        if not (AREA_MIN_DADO <= area <= AREA_MAX_DADO):          # filtro por área
            clase[k] = "ruido"
            continue

        # Recorte con margen alrededor del dado
        x0, y0 = max(x - PAD, 0), max(y - PAD, 0)
        x1, y1 = min(x + w + PAD, W), min(y + h + PAD, H)
        comp = (et[y0:y1, x0:x1] == k).astype(np.uint8) * 255

        # 4) Filtro por forma: cuadrado
        indices[k] = indice_cuadrado(comp)
        if indices[k] < CUADRADO_MIN:
            clase[k] = "descartada"
            continue

        # 5) Otsu usando SOLO los píxeles del dado (el fondo no entra al cálculo)
        interior = cv2.erode(comp, KERNEL_BORDE)                  # sin el borde del dado
        recorte_gris = gris[y0:y1, x0:x1]
        valores = recorte_gris[interior > 0].reshape(-1, 1)
        if valores.size == 0:
            clase[k] = "descartada"
            continue
        umbral, _ = cv2.threshold(valores, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        claros = ((recorte_gris > umbral) & (interior > 0)).astype(np.uint8) * 255

        # 6) Apertura sobre los puntos y conteo de blobs redondos
        puntos_bin = cv2.morphologyEx(claros, cv2.MORPH_OPEN, KERNEL_PUNTOS)
        valor, blobs, et_puntos = clasificar_puntos(puntos_bin)

        clase[k] = "dado"
        mu = cv2.moments(comp, binaryImage=True)
        dado = {"bbox": (x, y, w, h),
                "centro": (mu["m10"] / mu["m00"] + x0, mu["m01"] / mu["m00"] + y0),
                "valor": valor}
        if pasos is not None:
            dado["pasos"] = {"recorte_color": frame[y0:y1, x0:x1].copy(), "umbral": float(umbral),
                             "indice": indices[k], "claros": claros, "puntos_bin": puntos_bin,
                             "blobs": blobs, "et_puntos": et_puntos}
        dados.append(dado)

    # Ids únicos: orden de lectura (arriba -> abajo, izquierda -> derecha)
    dados.sort(key=lambda d: (d["centro"][1], d["centro"][0]))
    for i, d in enumerate(dados, 1):
        d["id"] = i

    if pasos is not None:
        pasos.update({"hsv": hsv, "rojo": rojo, "cerrada": cerrada, "abierta": abierta,
                      "etiquetas": et, "estadisticas": stats, "clase": clase, "indices": indices})
    return dados


# =====================================================================================
# C) VIDEO RESULTANTE
# =====================================================================================
def dibujar_dados(frame, dados):
    for d in dados:
        x, y, w, h = d["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_CAJA, 3)
        texto = f"D{d['id']}-{d['valor']}"
        pos_y = y - 12 if y > 45 else y + h + 38                  # arriba de la caja (o abajo si no entra)
        cv2.putText(frame, texto, (x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_CAJA, 3, cv2.LINE_AA)


def hay_rojo_en_caja(frame, bbox):
    """True si queda algún píxel rojo dentro de la bounding box (en el frame completo)."""
    x, y, w, h = bbox
    return cv2.countNonZero(mascara_roja(frame[y:y + h, x:x + w])) >= ROJO_MIN_EN_CAJA


def ultimo_frame_con_caja(path, dados, fin):
    """Para cada dado, último frame en el que su caja sigue en el video.
    Después del reposo (frame `fin`), la caja de un dado se mantiene mientras quede algún píxel rojo
    dentro de ella; en el primer frame sin rojo desaparece y no vuelve."""
    ultimo = [fin] * len(dados)
    activos = [True] * len(dados)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fin + 1)
    n = fin + 1
    while any(activos):
        ok, frame = cap.read()
        if not ok:
            break
        for i, d in enumerate(dados):
            if activos[i]:
                if hay_rojo_en_caja(frame, d["bbox"]):
                    ultimo[i] = n
                else:
                    activos[i] = False
        n += 1
    cap.release()
    return ultimo


def procesar_video(path_in, path_out):
    nombre = os.path.splitext(os.path.basename(path_in))[0]

    # A) Frames con los dados quietos
    res = rango_quieto(leer_mascaras(path_in))
    dados, ultimo, ini, fin = [], [], -1, -2
    if res is None:
        print(f"{nombre}: no se encontraron dados en ningún frame.")
    else:
        ref, ini, fin, _ = res

        # B) Valor de cada dado (una sola vez, en el frame de referencia)
        dados = detectar_dados(leer_frame(path_in, ref))
        print(f"{nombre}: dados quietos en los frames {ini}-{fin}")
        print("   " + ", ".join(f"D{d['id']}-{d['valor']}" for d in dados))
        if len(dados) != 5:
            print(f"   ATENCIÓN: se detectaron {len(dados)} dados (se esperaban 5)")
        ultimo = ultimo_frame_con_caja(path_in, dados, fin)

    # C) Video resultante: caja + id + valor desde el reposo y hasta que no queda rojo dentro de la caja
    cap = cv2.VideoCapture(path_in)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dibujar_dados(frame, [d for d, u in zip(dados, ultimo) if ini <= n <= u])
        out.write(frame)
        n += 1
    cap.release()
    out.release()


def main(carpeta):
    rutas = glob.glob(os.path.join(carpeta, "videos_input/tirada_*.mp4"))
    rutas.sort(key=lambda p: int(re.findall(r"\d+", os.path.splitext(os.path.basename(p))[0])[-1]))
    if not rutas:
        print(f"No se encontraron videos tirada_*.mp4 en {carpeta}")
        return
    for ruta in rutas:
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        procesar_video(ruta, os.path.join(carpeta, f"videos_output/resultado_{nombre}.mp4"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))

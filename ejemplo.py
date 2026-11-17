# -*- coding: utf-8 -*-
"""
Muestra, paso a paso, cómo se procesa una tirada: se abre un plot nuevo por cada paso.
Usa las mismas funciones que tp3_cinco_dados.py (deben estar en la misma carpeta), así que
lo que ves es lo que hace el código que genera los videos.

Uso:
    python pasos_tirada.py                 # tirada_1.mp4
    python pasos_tirada.py 3               # tirada_3.mp4
    python pasos_tirada.py ruta/video.mp4  # cualquier video

Plots (en orden):
    A1, A2) frames con los dados quietos
    1-7)    pipeline sobre el frame quieto: frame, HSV, rojos, clausura, apertura, componentes, resultado
    8-11)   por dado (un plot por paso, con todos los dados): recorte, Otsu, apertura, puntos detectados
    12)     cajas en el video: antes del reposo, durante, y mientras quede rojo dentro de cada caja
"""
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

import trabajo_practico as tp

MARGEN = 80      # margen (px) alrededor de los dados para recortar las imágenes


def elegir_video():
    carpeta = os.path.dirname(os.path.abspath(__file__))
    arg = sys.argv[1] if len(sys.argv) > 1 else "1"
    return arg if os.path.isfile(arg) else os.path.join(carpeta, f"tirada_{arg}.mp4")


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def plot(img, titulo, cmap=None, tam=(8, 8)):
    """Abre un plot nuevo con una sola imagen. Devuelve el eje (para poder anotar encima)."""
    fig, ax = plt.subplots(figsize=tam)
    ax.imshow(img, cmap=cmap)
    ax.set_title(titulo)
    ax.axis("off")
    fig.tight_layout()
    return ax


def plot_dados(imagenes, titulos, titulo_general, cmap=None):
    """Abre un plot nuevo con una imagen por dado, todos lado a lado."""
    n = len(imagenes)
    fig, axs = plt.subplots(1, n, figsize=(3.4 * n, 4.6), squeeze=False)
    for ax, img, t in zip(axs[0], imagenes, titulos):
        ax.imshow(img, cmap=cmap)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.suptitle(titulo_general, fontsize=12)
    fig.tight_layout()


def main():
    path = elegir_video()
    if not os.path.isfile(path):
        print(f"No existe el video: {path}")
        return
    nombre = os.path.basename(path)

    # ------------------------------------------------------------------ A) frames quietos
    mascaras = tp.leer_mascaras(path)
    res = tp.rango_quieto(mascaras)
    if res is None:
        print("No se encontraron dados en ningún frame.")
        return
    ref, ini, fin, dif = res
    n = len(mascaras)
    areas = np.array([cv2.countNonZero(m) for m in mascaras])
    cambios = np.array([0] + [cv2.countNonZero(cv2.bitwise_xor(mascaras[k], mascaras[k - 1]))
                              for k in range(1, n)])

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(range(n), cambios, color="tab:blue", width=0.9)
    ax.set_ylabel("px cambiados vs frame anterior")
    ax.set_xlabel("frame")
    ax.set_title(f"{nombre} - A1: cambio de la máscara roja entre frames consecutivos (barras) y área roja (línea)")
    ax2 = ax.twinx()
    ax2.plot(areas, color="tab:red", lw=1)
    ax2.set_ylabel("área roja (px)", color="tab:red")
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(range(n), np.minimum(dif, 1.0), color="tab:blue", width=0.9)
    ax.axhline(tp.UMBRAL, color="k", ls="--", lw=1, label=f"umbral = {tp.UMBRAL}")
    ax.axvspan(ini - 0.5, fin + 0.5, color="red", alpha=0.2, label=f"dados quietos: frames {ini}-{fin}")
    ax.axvline(ref, color="red", lw=1, label=f"frame de referencia: {ref}")
    ax.set_title(f"{nombre} - A2: diferencia contra el frame de referencia (solo píxeles rojos, tope en 1)")
    ax.set_xlabel("frame")
    ax.legend(loc="upper right")
    fig.tight_layout()

    # ------------------------------------------------------------------ pipeline sobre el frame quieto
    frame = tp.leer_frame(path, ref)
    pasos = {}
    dados = tp.detectar_dados(frame, pasos)
    H, W = frame.shape[:2]

    # Recorte alrededor de los dados para verlos más grandes
    if dados:
        x0 = max(min(d["bbox"][0] for d in dados) - MARGEN, 0)
        y0 = max(min(d["bbox"][1] for d in dados) - MARGEN, 0)
        x1 = min(max(d["bbox"][0] + d["bbox"][2] for d in dados) + MARGEN, W)
        y1 = min(max(d["bbox"][1] + d["bbox"][3] for d in dados) + MARGEN, H)
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    zona = (slice(y0, y1), slice(x0, x1))

    plot(rgb(frame)[zona], f"Paso 1: frame {ref} (dados quietos)")
    plot(pasos["hsv"][zona], "Paso 2: imagen en HSV (falso color)")
    plot(pasos["rojo"][zona], "Paso 3: píxeles rojos", "gray")
    plot(pasos["cerrada"][zona], f"Paso 4: clausura {tp.TAM_CIERRE}x{tp.TAM_CIERRE} (rellena los puntos)", "gray")
    plot(pasos["abierta"][zona], f"Paso 5: apertura {tp.TAM_APERTURA}x{tp.TAM_APERTURA} (erosión + dilatación)", "gray")

    # Paso 6: componentes conexas, coloreadas según lo que se decidió de cada una
    comp = np.zeros((H, W, 3), np.uint8)
    colores = {"dado": (0, 200, 0), "descartada": (220, 0, 0), "ruido": (110, 110, 110)}
    for k, clase in pasos["clase"].items():
        comp[pasos["etiquetas"] == k] = colores[clase]
    ax = plot(comp[zona], f"Paso 6: componentes conexas (número: índice cuadrado)\n"
                          f"verde: dado | rojo: no cuadrada | gris: área fuera de rango")
    for k, idx in pasos["indices"].items():
        bx, by, bw, bh, _ = [int(v) for v in pasos["estadisticas"][k]]
        ax.text(bx + bw / 2 - x0, by + bh / 2 - y0, f"{k}: {idx:.2f}", color="white", ha="center", va="center",
                fontsize=11, bbox=dict(facecolor="black", alpha=0.6, pad=2))

    anotado = frame.copy()
    tp.dibujar_dados(anotado, dados)
    plot(rgb(anotado)[zona], "Paso 7: resultado (caja, id y valor de cada dado)")

    # ------------------------------------------------------------------ por dado (un plot por paso)
    print(f"\n{nombre}: " + ", ".join(f"D{d['id']}-{d['valor']}" for d in dados))
    if dados:
        ids = [d["id"] for d in dados]
        plot_dados([rgb(d["pasos"]["recorte_color"]) for d in dados],
                   [f"D{i}" for i in ids],
                   "Paso 8: cada dado recortado")
        plot_dados([d["pasos"]["claros"] for d in dados],
                   [f"D{d['id']}: umbral {d['pasos']['umbral']:.0f}" for d in dados],
                   "Paso 9: Otsu (usando solo los píxeles de cada dado)", cmap="gray")
        plot_dados([d["pasos"]["puntos_bin"] for d in dados],
                   [f"D{i}" for i in ids],
                   f"Paso 10: apertura {tp.TAM_PUNTOS}x{tp.TAM_PUNTOS} sobre los puntos", cmap="gray")
        vistas = []
        for d in dados:
            p = d["pasos"]
            vis = np.zeros((*p["puntos_bin"].shape, 3), np.uint8)
            for b in p["blobs"]:
                color = (0, 200, 0) if b["ok"] else ((220, 0, 0) if b["ejes"] is not None else (110, 110, 110))
                vis[p["et_puntos"] == b["k"]] = color
            vistas.append(vis)
            print(f"  D{d['id']} (índice cuadrado {p['indice']:.2f}, umbral Otsu {p['umbral']:.0f}):")
            for b in p["blobs"]:
                ejes = "-" if b["ejes"] is None else f"{b['ejes']:.2f}"
                print(f"      blob área={b['area']:4d} | ejes menor/mayor={ejes} | {'PUNTO' if b['ok'] else 'descartado'}")
        plot_dados(vistas,
                   [f"D{d['id']}: {d['valor']} puntos" for d in dados],
                   "Paso 11: puntos detectados (verde: punto | rojo: no redondo | gris: área fuera de rango)")

    # ------------------------------------------------------------------ C) cajas en el video
    if dados:
        ultimo = tp.ultimo_frame_con_caja(path, dados, fin)
        print("\nÚltimo frame con caja de cada dado: " + ", ".join(f"D{d['id']}: {u}" for d, u in zip(dados, ultimo)))
        ult = max(ultimo)
        idxs = (max(ini - 1, 0), ref, (fin + ult) // 2, min(ult + 1, n - 1))
        imagenes, titulos = [], []
        for idx in idxs:
            f = tp.leer_frame(path, idx)
            visibles = [d for d, u in zip(dados, ultimo) if ini <= idx <= u]
            tp.dibujar_dados(f, visibles)
            imagenes.append(rgb(f)[zona])
            titulos.append(f"Frame {idx}: {len(visibles)} caja(s)")
        plot_dados(imagenes, titulos,
                   "Paso 12: las cajas aparecen con los dados quietos y se mantienen mientras quede rojo dentro de cada una")

    plt.show()


if __name__ == "__main__":
    main()

"""
Hand Control v2 — controla el mouse, dibuja en el aire y juega con tu mano.

Modos (cambian con MANO ABIERTA sostenida 1,5 s, o con las teclas m / d / j):
  MOUSE    -> el indice mueve el cursor, pellizco (pulgar + indice) = clic / arrastrar
  PIZARRA  -> pellizco = dibujar sobre el video; la barra de arriba se elige pasando el dedo
  JUEGO    -> revienta burbujas tocandolas con el indice (30 s)

Gestos:
  Puno cerrado 1 s      -> pausa / reanuda (el mouse queda libre)
  Mano abierta 1,5 s    -> siguiente modo
  Pulgar arriba 1,5 s   -> guardar dibujo (modo PIZARRA)

Teclas (con la ventana de la camara activa):
  q / Esc salir | m d j modos | c limpiar | s guardar | f pantalla completa
  h esqueleto de la mano | n numerar los 21 puntos | + / - grosor del trazo
Teclas globales (funcionan aunque la ventana no este activa, requiere `pip install pynput`):
  F8 pausa | F9 salir
"""

import math
import os
import random
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# =========================================================
# Configuracion
# =========================================================
pyautogui.FAILSAFE = True   # llevar el mouse REAL a una esquina = pausa de emergencia
# por defecto pyautogui espera 0,1 s tras cada llamada (bajaba los FPS)
pyautogui.PAUSE = 0

# suavizado (mayor = responde mas rapido, menor = mas suave)
EMA_ALPHA = 0.35
# recorte de bordes para el modo mouse (mas margen = mas precision)
WORK_AREA_MARGIN = 0.12

# pellizco se activa bajo este valor (relativo al tamano de la mano)
PINCH_ON = 0.30
# y se suelta sobre este (histeresis: evita clics fantasma)
PINCH_OFF = 0.42
PINCH_FRAMES = 2            # frames seguidos para confirmar el cambio

T_PAUSA = 1.0               # segundos de puno para pausar
T_MODO = 1.5                # segundos de mano abierta para cambiar modo
T_GUARDAR = 1.5             # segundos de pulgar arriba para guardar
T_BOTON = 0.6               # segundos sobre un boton de la barra para activarlo

DURACION_JUEGO = 30         # segundos por partida
CARPETA_SALIDAS = "salidas"
VENTANA = "Hand Control"

MODOS = ["MOUSE", "PIZARRA", "JUEGO"]

# Colores en BGR (OpenCV usa Azul, Verde, Rojo)
COLORES = {
    "Rojo": (40, 40, 230),
    "Naranjo": (0, 140, 255),
    "Amarillo": (0, 220, 255),
    "Verde": (60, 200, 60),
    "Azul": (230, 120, 30),
    "Morado": (200, 60, 160),
}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# =========================================================
# Teclas globales (opcional)
# =========================================================
estado_global = {"pausa_toggle": False, "salir": False}

try:
    from pynput import keyboard

    def _on_press(tecla):
        if tecla == keyboard.Key.f8:
            estado_global["pausa_toggle"] = True
        elif tecla == keyboard.Key.f9:
            estado_global["salir"] = True

    keyboard.Listener(on_press=_on_press, daemon=True).start()
    TECLAS_GLOBALES = True
except Exception:
    TECLAS_GLOBALES = False

# =========================================================
# Utilidades
# =========================================================


def dist(a, b):
    return math.dist((a.x, a.y), (b.x, b.y))


def suavizar(prev, nuevo, alpha=EMA_ALPHA):
    return nuevo if prev is None else (1 - alpha) * prev + alpha * nuevo


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def texto(img, msg, pos, escala=0.8, color=(255, 255, 255), fondo=(0, 0, 0), grosor=2, alpha=0.55):
    """Texto con fondo semitransparente (se lee mejor en el proyector).
    Ojo: cv2.putText no soporta tildes ni n con tilde."""
    (tw, th), base = cv2.getTextSize(
        msg, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    x, y = pos
    x0, y0, x1, y1 = x - 8, y - th - 8, x + tw + 8, y + base + 6
    x0, y0 = max(0, x0), max(0, y0)
    roi = img[y0:y1, x0:x1]
    if roi.size:
        capa = np.full_like(roi, fondo)
        cv2.addWeighted(capa, alpha, roi, 1 - alpha, 0, roi)
    cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                escala, color, grosor, cv2.LINE_AA)


def texto_centrado(img, msg, y, escala=1.5, color=(255, 255, 255), grosor=3):
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    texto(img, msg, ((img.shape[1] - tw) // 2, y),
          escala, color, grosor=grosor)


def abrir_camara():
    """Elige el backend segun el sistema operativo y prueba varios indices."""
    if sys.platform == "win32":
        backend = cv2.CAP_DSHOW
    elif sys.platform == "darwin":
        backend = cv2.CAP_AVFOUNDATION
    else:
        backend = cv2.CAP_ANY
    for indice in (0, 1, 2):
        cap = cv2.VideoCapture(indice, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"Camara abierta en indice {indice}")
            return cap
        cap.release()
    raise RuntimeError(
        "No se pudo abrir ninguna camara (probe indices 0, 1 y 2).")

# =========================================================
# Gestos
# =========================================================


def tamano_mano(lm):
    """Distancia muneca -> base del dedo medio. Sirve para que todo funcione
    igual estando cerca o lejos de la camara."""
    return max(dist(lm[0], lm[9]), 1e-6)


def dedos_arriba(lm):
    """Devuelve [pulgar, indice, medio, anular, menique] con True si esta estirado.
    Compara distancias (no posiciones), asi funciona aunque la mano este girada."""
    muneca = lm[0]
    dedos = [dist(lm[4], lm[17]) > dist(lm[3], lm[17])
             * 1.1]  # pulgar vs base del menique
    for punta, articulacion in ((8, 6), (12, 10), (16, 14), (20, 18)):
        dedos.append(dist(lm[punta], muneca) > dist(lm[articulacion], muneca))
    return dedos


def nombre_gesto(dedos):
    if not any(dedos):
        return "PUNO"
    if all(dedos):
        return "ABIERTA"
    if dedos == [True, False, False, False, False]:
        return "PULGAR"
    return None


class Pellizco:
    """Pellizco con histeresis + confirmacion por frames (evita clics fantasma)."""

    def __init__(self):
        self.activo = False
        self.contador = 0

    def actualizar(self, razon):
        quiere_cambiar = (razon < PINCH_ON) if not self.activo else (
            razon > PINCH_OFF)
        self.contador = self.contador + 1 if quiere_cambiar else 0
        if self.contador >= PINCH_FRAMES:
            self.activo = not self.activo
            self.contador = 0
        return self.activo

    def reset(self):
        self.activo = False
        self.contador = 0


class GestoSostenido:
    """Mide cuanto tiempo se mantiene un gesto. Dispara una sola vez al completarse."""

    def __init__(self):
        self.gesto = None
        self.inicio = 0.0
        self.disparado = False

    def actualizar(self, gesto, duraciones):
        """Devuelve (gesto, progreso 0..1, disparo True/False)."""
        ahora = time.time()
        if gesto != self.gesto:
            self.gesto, self.inicio, self.disparado = gesto, ahora, False
        if gesto not in duraciones:
            return gesto, 0.0, False
        progreso = min(1.0, (ahora - self.inicio) / duraciones[gesto])
        if progreso >= 1.0 and not self.disparado:
            self.disparado = True
            return gesto, 1.0, True
        return gesto, (0.0 if self.disparado else progreso), False

# =========================================================
# Modo PIZARRA
# =========================================================


class Pizarra:
    ALTO_BARRA = 80

    def __init__(self, w, h):
        self.lienzo = np.zeros((h, w, 3), dtype=np.uint8)  # negro = sin tinta
        self.color = COLORES["Azul"]
        self.nombre_color = "Azul"
        self.grosor = 8
        self.prev = None
        self.hover = None
        self.hover_inicio = 0.0
        self.hover_usado = False
        nombres = list(COLORES) + ["Borrar", "Limpiar", "Guardar"]
        ancho = w // len(nombres)
        self.botones = [(n, i * ancho, (i + 1) * ancho)
                        for i, n in enumerate(nombres)]

    def limpiar(self):
        self.lienzo[:] = 0

    def guardar(self):
        os.makedirs(CARPETA_SALIDAS, exist_ok=True)
        blanco = np.full_like(self.lienzo, 255)
        mascara = self.lienzo.any(axis=2)
        blanco[mascara] = self.lienzo[mascara]
        nombre = os.path.join(
            CARPETA_SALIDAS, f"dibujo_{int(time.time())}.png")
        cv2.imwrite(nombre, blanco)
        print("Guardado:", nombre)
        return nombre

    def _activar(self, boton):
        if boton in COLORES:
            self.color, self.nombre_color = COLORES[boton], boton
        elif boton == "Borrar":
            self.color, self.nombre_color = (0, 0, 0), "Borrar"
        elif boton == "Limpiar":
            self.limpiar()
        elif boton == "Guardar":
            return "guardar"
        return None

    def actualizar(self, punto, dibujando):
        """punto = (x, y) del indice en pixeles, o None si no hay mano."""
        if punto is None:
            self.prev, self.hover = None, None
            return None
        x, y = punto
        # ¿Esta el dedo sobre la barra?
        if y < self.ALTO_BARRA:
            self.prev = None
            boton = next(
                (n for n, x0, x1 in self.botones if x0 <= x < x1), None)
            if boton != self.hover:
                self.hover, self.hover_inicio, self.hover_usado = boton, time.time(), False
            elif not self.hover_usado and time.time() - self.hover_inicio >= T_BOTON:
                self.hover_usado = True
                return self._activar(boton)
            return None
        self.hover = None
        if dibujando:
            if self.prev is not None:
                grosor = self.grosor * 4 if self.nombre_color == "Borrar" else self.grosor
                cv2.line(self.lienzo, self.prev, punto,
                         self.color, grosor, cv2.LINE_AA)
            self.prev = punto
        else:
            self.prev = None
        return None

    def dibujar(self, frame, punto):
        # Tinta sobre el video
        mascara = self.lienzo.any(axis=2)
        frame[mascara] = self.lienzo[mascara]
        # Barra de botones
        for nombre, x0, x1 in self.botones:
            color = COLORES.get(nombre, (70, 70, 70))
            cv2.rectangle(frame, (x0 + 4, 4),
                          (x1 - 4, self.ALTO_BARRA - 4), color, -1)
            if nombre == self.nombre_color:
                cv2.rectangle(frame, (x0 + 4, 4), (x1 - 4,
                              self.ALTO_BARRA - 4), (255, 255, 255), 4)
            if nombre == self.hover and not self.hover_usado:
                p = min(1.0, (time.time() - self.hover_inicio) / T_BOTON)
                cv2.rectangle(frame, (x0 + 4, self.ALTO_BARRA - 12),
                              (x0 + 4 + int((x1 - x0 - 8) * p), self.ALTO_BARRA - 4), (255, 255, 255), -1)
            cv2.putText(frame, nombre, (x0 + 10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
        # Puntero: muestra color y grosor actuales
        if punto is not None:
            borde = (255, 255, 255) if self.nombre_color != "Borrar" else (
                0, 0, 255)
            cv2.circle(frame, punto, max(self.grosor, 6), borde, 2)

# =========================================================
# Modo JUEGO: revienta burbujas
# =========================================================


class Juego:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.record = 0
        self.estado = "inicio"  # inicio | jugando | fin
        self.burbujas = []
        self.explosiones = []
        self.puntaje = 0

    def empezar(self):
        self.estado = "jugando"
        self.puntaje = 0
        self.burbujas, self.explosiones = [], []
        self.inicio = time.time()
        self.proxima = 0.0

    def tiempo_restante(self):
        return max(0.0, DURACION_JUEGO - (time.time() - self.inicio))

    def actualizar(self, punto, pellizco_nuevo):
        if self.estado in ("inicio", "fin"):
            if pellizco_nuevo:
                self.empezar()
            return
        ahora = time.time()
        avance = 1 - self.tiempo_restante() / DURACION_JUEGO  # 0 -> 1 durante la partida
        # Aparecen burbujas cada vez mas seguido, mas rapidas y mas chicas
        if ahora >= self.proxima:
            r = int(random.uniform(55, 75) - 25 * avance)
            self.burbujas.append({
                "x": random.randint(r, self.w - r), "y": self.h + r, "r": r,
                "v": random.uniform(3, 5) + 6 * avance,
                "color": random.choice(list(COLORES.values())),
            })
            self.proxima = ahora + max(0.25, 0.9 - 0.65 * avance)
        for b in self.burbujas:
            b["y"] -= b["v"]
        # ¿El dedo toco alguna?
        if punto is not None:
            for b in self.burbujas[:]:
                if math.dist(punto, (b["x"], b["y"])) < b["r"]:
                    self.burbujas.remove(b)
                    self.explosiones.append(
                        {"x": b["x"], "y": b["y"], "r": b["r"], "t": ahora, "c": b["color"]})
                    self.puntaje += 1
        self.burbujas = [b for b in self.burbujas if b["y"] > -b["r"]]
        self.explosiones = [
            e for e in self.explosiones if ahora - e["t"] < 0.3]
        if self.tiempo_restante() <= 0:
            self.estado = "fin"
            self.record = max(self.record, self.puntaje)

    def dibujar(self, frame):
        if self.estado == "inicio":
            texto_centrado(frame, "REVIENTA BURBUJAS",
                           self.h // 2 - 40, 1.8, (0, 220, 255))
            texto_centrado(frame, "Pellizca para empezar",
                           self.h // 2 + 30, 1.0)
            return
        for b in self.burbujas:
            centro = (int(b["x"]), int(b["y"]))
            cv2.circle(frame, centro, b["r"], b["color"], -1, cv2.LINE_AA)
            cv2.circle(frame, centro, b["r"], (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(frame, (centro[0] - b["r"] // 3, centro[1] - b["r"] // 3), b["r"] // 6,
                       (255, 255, 255), -1, cv2.LINE_AA)
        for e in self.explosiones:
            p = (time.time() - e["t"]) / 0.3
            cv2.circle(frame, (int(e["x"]), int(e["y"])), int(
                e["r"] * (1 + p)), e["c"], 4, cv2.LINE_AA)
        texto(frame, f"Puntaje: {self.puntaje}", (20, 50), 1.2, (0, 255, 0))
        texto(frame, f"Tiempo: {self.tiempo_restante():.0f}",
              (self.w - 230, 50), 1.2, (0, 220, 255))
        if self.estado == "fin":
            texto_centrado(frame, "SE ACABO EL TIEMPO",
                           self.h // 2 - 70, 1.6, (0, 220, 255))
            texto_centrado(
                frame, f"Puntaje: {self.puntaje}   Record: {self.record}", self.h // 2, 1.3)
            texto_centrado(frame, "Pellizca para jugar otra vez",
                           self.h // 2 + 60, 1.0)

# =========================================================
# Programa principal
# =========================================================


def dibujar_anillo(frame, centro, progreso, color):
    """Anillo de progreso alrededor de la mano mientras se sostiene un gesto."""
    if progreso <= 0:
        return
    cv2.ellipse(frame, centro, (70, 70), -90, 0,
                360, (60, 60, 60), 6, cv2.LINE_AA)
    cv2.ellipse(frame, centro, (70, 70), -90, 0,
                int(360 * progreso), color, 8, cv2.LINE_AA)


def main():
    screen_w, screen_h = pyautogui.size()
    cap = abrir_camara()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("La camara no entrega imagen.")
    h, w = frame.shape[:2]

    hands = mp_hands.Hands(max_num_hands=1, model_complexity=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.6)

    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA, w, h)

    modo = "MOUSE"
    pausado = False
    pantalla_completa = False
    ver_esqueleto = True
    ver_numeros = False
    mouse_abajo = False
    sx = sy = None                      # posicion suavizada del indice (0..1)
    pellizco = Pellizco()
    sostenido = GestoSostenido()
    pizarra = Pizarra(w, h)
    juego = Juego(w, h)
    aviso, aviso_hasta = "", 0.0
    fps, t_prev = 0.0, time.time()

    def avisar(msg, segundos=1.5):
        nonlocal aviso, aviso_hasta
        aviso, aviso_hasta = msg, time.time() + segundos

    def soltar_mouse():
        nonlocal mouse_abajo
        if mouse_abajo:
            try:
                pyautogui.mouseUp()
            except pyautogui.FailSafeException:
                pass
            mouse_abajo = False

    def cambiar_modo(nuevo):
        nonlocal modo
        soltar_mouse()
        pellizco.reset()
        pizarra.prev = None
        modo = nuevo
        if modo == "JUEGO":
            juego.estado = "inicio"
        avisar(f"Modo {modo}")

    print(__doc__)
    if not TECLAS_GLOBALES:
        print("(Aviso) pynput no esta instalado: F8/F9 no funcionaran. pip install pynput")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # --- Teclas globales (pynput) ---
        if estado_global["salir"]:
            break
        if estado_global["pausa_toggle"]:
            estado_global["pausa_toggle"] = False
            pausado = not pausado
            soltar_mouse()
            avisar("PAUSADO" if pausado else "Reanudado")

        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        punto = None            # punta del indice en pixeles del video
        razon = None
        progreso, color_anillo, centro_mano = 0.0, (255, 255, 255), None
        pellizco_antes = pellizco.activo

        if res.multi_hand_landmarks:
            mano = res.multi_hand_landmarks[0]
            lm = mano.landmark
            tam = tamano_mano(lm)
            razon = dist(lm[4], lm[8]) / tam
            dedos = dedos_arriba(lm)
            gesto = nombre_gesto(dedos)
            pellizcando = pellizco.actualizar(razon)

            sx = suavizar(sx, lm[8].x)
            sy = suavizar(sy, lm[8].y)
            punto = (int(clamp(sx, 0, 1) * (w - 1)),
                     int(clamp(sy, 0, 1) * (h - 1)))
            centro_mano = (int(lm[9].x * w), int(lm[9].y * h))

            if ver_esqueleto:
                mp_draw.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)
            if ver_numeros:
                for i, p in enumerate(lm):
                    cv2.putText(frame, str(i), (int(p.x * w) + 5, int(p.y * h) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            # --- Gestos sostenidos ---
            duraciones = {"PUNO": T_PAUSA, "ABIERTA": T_MODO}
            if modo == "PIZARRA":
                duraciones["PULGAR"] = T_GUARDAR
            g, progreso, disparo = sostenido.actualizar(gesto, duraciones)
            color_anillo = {"PUNO": (0, 0, 255), "ABIERTA": (
                0, 220, 255), "PULGAR": (0, 255, 0)}.get(g, (255, 255, 255))
            if disparo:
                if g == "PUNO":
                    pausado = not pausado
                    soltar_mouse()
                    avisar("PAUSADO" if pausado else "Reanudado")
                elif g == "ABIERTA" and not pausado:
                    cambiar_modo(MODOS[(MODOS.index(modo) + 1) % len(MODOS)])
                elif g == "PULGAR" and not pausado:
                    pizarra.guardar()
                    avisar("Dibujo guardado!")

            # --- Acciones segun el modo ---
            if not pausado and modo == "MOUSE":
                if gesto != "PUNO":  # con el puno el cursor se queda quieto
                    mx = clamp((sx - WORK_AREA_MARGIN) /
                               (1 - 2 * WORK_AREA_MARGIN), 0, 1)
                    my = clamp((sy - WORK_AREA_MARGIN) /
                               (1 - 2 * WORK_AREA_MARGIN), 0, 1)
                    cx = int(clamp(mx * screen_w, 2, screen_w - 3))
                    cy = int(clamp(my * screen_h, 2, screen_h - 3))
                    try:
                        pyautogui.moveTo(cx, cy)
                        if pellizcando and not mouse_abajo:
                            pyautogui.mouseDown()
                            mouse_abajo = True
                        elif not pellizcando and mouse_abajo:
                            pyautogui.mouseUp()
                            mouse_abajo = False
                    except pyautogui.FailSafeException:
                        # Alguien llevo el mouse real a una esquina: pausa de emergencia
                        pausado = True
                        mouse_abajo = False
                        avisar("PAUSA DE EMERGENCIA (esquina)", 2.5)
            elif not pausado and modo == "PIZARRA":
                if pizarra.actualizar(punto, pellizcando) == "guardar":
                    pizarra.guardar()
                    avisar("Dibujo guardado!")
        else:
            # Sin mano: soltar todo
            soltar_mouse()
            pellizco.reset()
            sostenido.actualizar(None, {})
            pizarra.actualizar(None, False)
            sx = sy = None

        if modo == "JUEGO" and not pausado:
            juego.actualizar(punto, pellizco.activo and not pellizco_antes)

        # --- Dibujo de la pantalla ---
        if modo == "MOUSE":
            x0, y0 = int(WORK_AREA_MARGIN * w), int(WORK_AREA_MARGIN * h)
            cv2.rectangle(frame, (x0, y0), (w - x0, h - y0),
                          (120, 120, 120), 1)
            if punto:
                cv2.circle(frame, punto, 10, (0, 255, 0)
                           if pellizco.activo else (0, 255, 255), -1)
        elif modo == "PIZARRA":
            pizarra.dibujar(frame, punto)
        elif modo == "JUEGO":
            juego.dibujar(frame)
            if punto:
                cv2.circle(frame, punto, 14, (255, 255, 255), 3)

        if centro_mano:
            dibujar_anillo(frame, centro_mano, progreso, color_anillo)

        # HUD inferior
        ahora = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(ahora - t_prev, 1e-6))
        t_prev = ahora
        estado = "PAUSADO" if pausado else modo
        texto(frame, f"{estado}   FPS {fps:.0f}", (15, h - 20), 0.9,
              (0, 0, 255) if pausado else (255, 255, 0))
        if razon is not None:  # barra del pellizco
            bx, by, bw = w - 260, h - 40, 240
            llenado = clamp(1 - razon / 1.0, 0, 1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + 20),
                          (60, 60, 60), -1)
            cv2.rectangle(frame, (bx, by), (bx + int(bw * llenado), by + 20),
                          (0, 255, 0) if pellizco.activo else (0, 200, 255), -1)
            umbral_x = bx + int(bw * (1 - PINCH_ON))
            cv2.line(frame, (umbral_x, by - 4),
                     (umbral_x, by + 24), (255, 255, 255), 2)
            cv2.putText(frame, "pellizco", (bx, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        if modo != "PIZARRA" and not (modo == "JUEGO" and juego.estado == "jugando"):
            ayuda = ["Puno 1s: pausa", "Mano abierta: modo",
                     "Pellizco: clic/dibujar"]
            for i, linea in enumerate(ayuda):
                texto(frame, linea, (w - 300, 40 + i * 38), 0.65)
        if pausado:
            texto_centrado(frame, "PAUSADO", h // 2, 2.5, (0, 0, 255), 5)
        if ahora < aviso_hasta:
            texto_centrado(frame, aviso, h // 2 +
                           (90 if pausado else 0), 1.4, (0, 255, 0))

        cv2.imshow(VENTANA, frame)

        # --- Teclado (ventana activa) ---
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("m"):
            cambiar_modo("MOUSE")
        elif k == ord("d"):
            cambiar_modo("PIZARRA")
        elif k == ord("j"):
            cambiar_modo("JUEGO")
        elif k == ord("p"):
            pausado = not pausado
            soltar_mouse()
        elif k == ord("c"):
            pizarra.limpiar()
        elif k == ord("s"):
            pizarra.guardar()
            avisar("Dibujo guardado!")
        elif k in (ord("+"), ord("=")):
            pizarra.grosor = min(40, pizarra.grosor + 2)
        elif k == ord("-"):
            pizarra.grosor = max(2, pizarra.grosor - 2)
        elif k == ord("h"):
            ver_esqueleto = not ver_esqueleto
        elif k == ord("n"):
            ver_numeros = not ver_numeros
        elif k == ord("f"):
            pantalla_completa = not pantalla_completa
            cv2.setWindowProperty(VENTANA, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if pantalla_completa else cv2.WINDOW_NORMAL)

    soltar_mouse()
    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

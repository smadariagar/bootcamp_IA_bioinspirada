#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOOTCAMP DE IA BIOINSPIRADA: "Aprendizaje desde propiedades locales"
Código Base Modular: Autómata de Conway + Vehículo de Braitenberg
====================================================================

Estructura diseñada para el trabajo interdisciplinario (Ingeniería + Biología):
  - Día 1 (Mañana): De Conway clásico (B3/S23) a reglas probabilísticas.
  - Día 1 (Tarde) : Inyección sensorial, decodificación motora y visualizaciones
                    neurocientíficas (Raster Plot y Tasa de Disparo Poblacional).
  - Día 2         : Restricciones biológicas locales (ATP/energía, inhibición,
                    períodos refractarios y plasticidad Hebbiana local).
  - Día 3         : Evaluación de desempeño (--headless) en arena cuadrada o circular.

Uso:
    python3 bootcamp_conway_carro.py
    python3 bootcamp_conway_carro.py --mode conway_prob --arena circular
    python3 bootcamp_conway_carro.py --headless 40000 --mode bio_plastico
"""

import argparse
import math
import sys
import threading
from decimal import Decimal, ROUND_HALF_UP

import numpy as np


# =====================================================================
# 1. PRNG DETERMINISTA (Réplica bit a bit de mulberry32 para evaluación)
# =====================================================================
MASK32 = 0xFFFFFFFF


class Mulberry32:
    """Generador pseudoaleatorio determinista con semilla de 32 bits."""
    __slots__ = ("a",)

    def __init__(self, a):
        self.a = a & MASK32

    def __call__(self):
        a = (self.a + 0x6D2B79F5) & MASK32
        self.a = a
        t = ((a ^ (a >> 15)) * (1 | a)) & MASK32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & MASK32)) & MASK32) ^ t
        return ((t ^ (t >> 14)) & MASK32) / 4294967296.0

    def random_matrix(self, shape):
        """Genera una matriz de floats en [0, 1) usando la semilla controlada."""
        flat = np.empty(int(np.prod(shape)), dtype=np.float32)
        for i in range(flat.size):
            flat[i] = self()
        return flat.reshape(shape)


def js_round(x):
    return math.floor(x + 0.5)


def tofixed(v, n):
    if v != v:
        return "NaN"
    q = Decimal(1).scaleb(-n)
    return str(Decimal(float(v)).quantize(q, rounding=ROUND_HALF_UP))


def pad(s, n):
    return str(s).rjust(n)


# =====================================================================
# 2. CONSTANTES DE LA GRILLA, SENSORES Y ARENA FÍSICA
# =====================================================================
N = 40                     # Grilla N x N (1600 unidades)
M = N * N

# Parámetros del modelo metabólico avanzado de referencia (40k)
P_EXC = 0.85
COST_MIN = 1.0 / 30.0
COST_MAX = 3.0 / 30.0
E_INIT = 300.0
E_MAX = 300.0
FAC_INIT = 2
COST_STEP = 0.1 / 40.0
NEIGH_HIGH = 10
E_SENSE_BOOST = 45.0
ADJ_RANGE = 30.0
ADJ_MOD = 5
ACT_THRESH = 2.0
W_INIT = 1.0
HEBB_LR = 0.05

# Parámetros físicos de la Arena y el Vehículo
PLANE = 1000
SENSOR_ANGLES = [d * math.pi / 180.0 for d in (0, 30, 45, 90, -30, -45, -90)]
NRAY = 7
MAXR = 160
RAYSTEP = 1
CAR_R = 10
CONTACT = 18
CRASH_RESET = 6
TRAMO = 5000

# Geometría circular opcional (de carro_evitación_40k_circular.py)
CENTRO_ARENA = np.array([500.0, 500.0], dtype=np.float64)
RADIO_ARENA = 490.0
RADIO_OBSTACULO = 35.0
OBSTACULOS_CIRCULARES = np.array([
    [745.0, 500.5],
    [624.4, 712.7],
    [373.9, 711.7],
    [255.0, 500.5],
    [376.0, 288.3],
    [625.6, 289.3],
], dtype=np.float64)

# Coordenadas espaciales de la grilla 40x40 (mundo 0..30)
SP = ADJ_RANGE / (N - 1)
_gx, _gy = np.meshgrid(np.arange(N), np.arange(N))
PXc = (_gx.astype(np.float64) * SP).astype(np.float32).ravel()
PYc = (_gy.astype(np.float64) * SP).astype(np.float32).ravel()

# Elipses sensoras por defecto: [cx, cy, indiceRayo, a2, b2, inhib]
SENSOR_ELLIPSES = [
    (26.0,  3.5, 1, 6.0, 15.0, 0),   # +30° -> Zona Derecha
    (26.0,  7.5, 2, 6.0, 15.0, 0),   # +45° -> Zona Derecha
    (26.0, 11.5, 3, 6.0, 15.0, 0),   # +90° -> Zona Derecha
    (4.0,  18.5, 4, 6.0, 15.0, 0),   # -30° -> Zona Izquierda
    (4.0,  22.5, 5, 6.0, 15.0, 0),   # -45° -> Zona Izquierda
    (4.0,  26.5, 6, 6.0, 15.0, 0),   # -90° -> Zona Izquierda
    (19.0,  7.5, 0, 6.0,  9.0, 1),   #   0° (frontal inhib) en Zona Derecha
    (11.0, 22.5, 0, 6.0,  9.0, 1),   #   0° (frontal inhib) en Zona Izquierda
]

# Historial para visualizaciones neurocientíficas (Raster Plot + Firing Rate)
HIST_LEN = 220
N_RASTER_NEURONS = 100


# =====================================================================
# 3. CLASE PRINCIPAL DE SIMULACIÓN (FÍSICA + ZONA DE TALLER EDITABLE)
# =====================================================================
class BootcampSim:
    """
    Simulador base del Bootcamp.
    Los participantes pueden editar directamente los métodos de la sección
    'ZONA DE TALLER' o crear una subclase que herede de BootcampSim.
    """

    def __init__(self):
        self.seedValue = 12345
        self.rng = Mulberry32(self.seedValue)

        # --- Selector de Modo del Autómata (Progresión Día 1 -> Día 3) ---
        # 'conway_clasico' : Juego de la Vida determinista B3/S23 (Vecindario Moore)
        # 'conway_prob'    : Día 1 - Reglas probabilísticas + ruido espontáneo + refractariedad
        # 'bio_plastico'   : Día 2 - Energía (ATP) + Inhibición + Plasticidad Hebbiana local
        # 'metabolico_40k' : Modelo avanzado de referencia (grafo disperso metabólico)
        self.brainMode = 'conway_prob'

        # --- Hiperparámetros para Día 1 y Día 2 (Conway Probabilístico y Bio-Plástico) ---
        self.pBirth = 0.90          # Prob. de nacer con 3 vecinos activos
        self.pSurvive = 0.95        # Prob. de sobrevivir con 2 o 3 vecinos
        self.pSpontaneous = 0.002   # Ruido basal (disparo espontáneo)
        self.refracSteps = 1        # Pasos de período refractario tras disparar
        self.atpDecay = 25.0        # Gasto de ATP por disparo (modo bio_plastico)
        self.atpRecovery = 4.0      # Recuperación de ATP en reposo (modo bio_plastico)
        self.localHebbRate = 0.02   # Tasa de ajuste de umbral local por co-activación

        # --- Parámetros del modelo metabólico 40k original ---
        self.ADJ_DIST = 11
        self.inhRadius = 6
        self.sparsity = 'modular'
        self.veto = True
        self.vetoK = 3
        self.inhCost = 0.5
        self.costMult = 1
        self.seedFrac = 0.35
        self.hetero = False
        self.hetSev = 1.0
        self.hebb = False
        self.sensBonus = 2.0
        self.sensWin = 2
        self.nVetoed = 0

        # --- Configuración de Arena y Vehículo ---
        self.arenaShape = 'cuadrada'   # 'cuadrada' o 'circular'
        self.fieldType = 'mixto'
        self.objSEP = 70
        self.carSpeed = 14.0
        self.senseR = 2
        self.sensGain = 3.0
        self.fbMode = 'distancia'
        self.sensorMode = 'fijas'
        self.Kchosen = 300
        self.frontInhib = True
        self.wiring = 'ipsi'           # 'ipsi' (evitación) o 'contra' (agresión)
        self.sensFeedback = True
        self.excEllipses = True
        self.pureCA = False
        self.controller = 'automata'
        self.moveEvery = 1
        self.bananaWarm = 0

        # --- Estado de simulación ---
        self.step = 0
        self.moveCounter = 0
        self.crashObj = 0
        self.crashBorder = 0
        self.distTraveled = 0.0
        self.crashesConsec = 0
        self.tramos = []
        self.tramoPrev = 0
        self.trail = []
        self.flash = 0
        self.flashColor = '#fff'
        self.lastDist = [float(MAXR)] * NRAY
        self.lastU = 0.0
        self.lastW = 0.0
        self.lastF1 = 0.0
        self.lastF2 = 0.0
        self.heldU = 0.0
        self.heldW = 0.0
        self.oscP = 0.0
        self.levyRem = 0
        self.levyTurn = 0.0
        self.lastExtStim = np.zeros(M, np.uint8)

        # --- Buffers para Visualizaciones Neurocientíficas ---
        self.rasterIndices = np.zeros(N_RASTER_NEURONS, dtype=np.int32)
        self.rasterTypes = np.zeros(N_RASTER_NEURONS, dtype=np.uint8)
        self.rasterHist = np.zeros((N_RASTER_NEURONS, HIST_LEN), dtype=np.uint8)
        self.rateGlobalHist = np.zeros(HIST_LEN, dtype=np.float32)
        self.rateDerHist = np.zeros(HIST_LEN, dtype=np.float32)
        self.rateIzqHist = np.zeros(HIST_LEN, dtype=np.float32)

        # --- Arena ---
        self.obstacle = np.zeros((PLANE, PLANE), np.uint8)
        self.obs_x = np.zeros(0, np.int32)
        self.obs_y = np.zeros(0, np.int32)
        self._obs_list = []
        self._obs_dirty = True
        self.bananas = []
        self.bananasTotal = 0
        self.bananasCollected = 0
        self.bananaArmed = False
        self.measStart = 0

        self._ds = np.arange(RAYSTEP, MAXR + 1, RAYSTEP, dtype=np.float64)
        self._dilated = None

        self.build_adjacency()
        self.reset()

    # =================================================================
    # ZONA DE TALLER (MÉTODOS EDITABLES POR LOS EQUIPOS EN DÍAS 1, 2 Y 3)
    # =================================================================

    def definir_zonas_sensoriomotrices(self):
        """
        [DÍA 1 - TARDE] Define qué celdas de la grilla 40x40 reciben estímulos
        sensoriales y qué regiones actúan como núcleos motores (rueda der/izq).
        """
        self.neuronaEnSensor = np.full(M, -1, np.int8)
        self.sensorInhib = np.zeros(M, np.uint8)
        self.sensorNeurons = [[] for _ in range(NRAY)]
        self.maskIzq = np.zeros(M, np.uint8)
        self.maskDer = np.zeros(M, np.uint8)

        for i in range(M):
            px = float(PXc[i])
            py = float(PYc[i])
            for (cx, cy, ray, a2, b2, inh) in SENSOR_ELLIPSES:
                dx = px - cx
                dy = py - cy
                if (dx * dx / a2 + dy * dy / b2) <= 1.0:
                    self.neuronaEnSensor[i] = ray
                    self.sensorInhib[i] = inh
            r = int(self.neuronaEnSensor[i])
            if r >= 0:
                self.sensorNeurons[r].append(i)
            # Regiones motoras por defecto: dos cuadrados de 14x14
            if 0.5 <= px <= 14.5 and 15.5 <= py <= 29.5:
                self.maskIzq[i] = 1
            if 15.5 <= px <= 29.5 and 0.5 <= py <= 14.5:
                self.maskDer[i] = 1

        self.nIzq = int(self.maskIzq.sum())
        self.nDer = int(self.maskDer.sum())
        self.maskIzqB = self.maskIzq.astype(bool)
        self.maskDerB = self.maskDer.astype(bool)
        self._rayLists = [np.array(l, np.int32) for l in self.sensorNeurons]

    def init_condiciones_locales(self):
        """
        [DÍA 1 MAÑANA Y DÍA 2] Condiciones iniciales de la grilla:
        tipos celulares (excitatorias/inhibitorias), reservas de energía (ATP),
        períodos refractarios, umbrales locales y semillas activas iniciales.
        """
        rng = self.rng
        tipo = np.zeros(M, np.uint8)
        costo = np.zeros(M, np.float32)
        for i in range(M):
            t = 1 if rng() < P_EXC else 0
            tipo[i] = t
            mg = COST_MIN + rng() * (COST_MAX - COST_MIN)
            costo[i] = mg if t == 1 else -mg
        self.tipo = tipo
        self.costo = costo
        self.energia = np.full(M, E_INIT, np.float32)
        self.guardada = np.zeros(M, np.float32)
        self.facilitado = np.full(M, FAC_INIT, np.uint8)

        # Matrices auxiliares para Conway Probabilístico y Bio-Plástico (2D: N x N)
        self.refracGrid = np.zeros((N, N), dtype=np.int32)
        self.umbralLocal = np.full((N, N), 3.0, dtype=np.float32)

        if self.excEllipses:
            m = (self.neuronaEnSensor >= 0) & (self.sensorInhib == 0)
            self.tipo[m] = 1
            self.costo[m] = np.abs(self.costo[m])

        self.w = np.full(self.src.size, W_INIT, np.float32)
        self.sens = np.zeros(M, np.uint8)
        self.grid = np.zeros(M, np.uint8)

        # Semillas activas al inicio
        tg = js_round(M * self.seedFrac)
        idx = list(range(M))
        for i in range(M - 1, 0, -1):
            j = int(rng() * (i + 1))
            idx[i], idx[j] = idx[j], idx[i]
        if tg > 0:
            self.grid[np.array(idx[:tg], np.int64)] = 1

        exc = self.tipo == 1
        self.nDerE = int(np.count_nonzero(exc & self.maskDerB))
        self.nIzqE = int(np.count_nonzero(exc & self.maskIzqB))
        self._refresh_edge_masks()
        self._init_neuro_recording()

    def codificar_sensores(self):
        """
        [DÍA 1 - TARDE] Inyección sensorial (Input):
        Transforma las distancias de los 7 sensores de rayos en una máscara
        de estimulación externa `extStim` (1 = forzar activación, 0 = sin estímulo).
        """
        extStim = np.zeros(M, np.uint8)
        sVals = [0.0] * NRAY
        for a in range(NRAY):
            d = self.ray_dist(self.carX, self.carY, self.carDir + SENSOR_ANGLES[a])
            self.lastDist[a] = d
            if self.fbMode == 'contacto':
                sVals[a] = 1.0 if d < CONTACT * self.sensGain else 0.0
            else:
                sVals[a] = max(0.0, min(1.0, (1.0 - d / MAXR) * self.sensGain))

        if not self.sensFeedback:
            return extStim

        rng = self.rng
        chosen = self.chosen
        tipo = self.tipo
        fijas = (self.sensorMode == 'fijas')
        for a in range(NRAY):
            p = sVals[a]
            if p <= 0:
                continue
            isFront = (a == 0)
            if isFront and not self.frontInhib:
                continue
            for n in self.sensorNeurons[a]:
                if isFront:
                    if tipo[n] != 0:
                        continue
                elif fijas and not chosen[n]:
                    continue
                if rng() < p:
                    extStim[n] = 1
        return extStim

    @staticmethod
    def _contar_vecinos_moore(G):
        """
        Calcula la suma de los 8 vecinos del vecindario de Moore en 2D
        con condiciones de borde cerradas (cero padding), 100% en NumPy.
        """
        P = np.pad(G.astype(np.float32), 1, mode='constant', constant_values=0)
        return (
            P[:-2, :-2] + P[:-2, 1:-1] + P[:-2, 2:] +
            P[1:-1, :-2]               + P[1:-1, 2:] +
            P[2:, :-2]  + P[2:, 1:-1]  + P[2:, 2:]
        )

    def paso_automata_local(self, t, extStim):
        """
        [DÍA 1 Y DÍA 2] Regla de actualización puramente local del autómata.
        Despacha según `self.brainMode`:
          1. 'conway_clasico' : Regla determinista de Conway B3/S23.
          2. 'conway_prob'    : Conway estocástico + refractariedad (Día 1).
          3. 'bio_plastico'   : Conway con ATP, inhibición y plasticidad Hebbiana (Día 2).
          4. 'metabolico_40k' : Grafo disperso metabólico de referencia.
        """
        if self.brainMode == 'metabolico_40k':
            self._step_metabolico_40k(t, extStim)
            return

        G = self.grid.reshape(N, N)
        ext2D = (extStim != 0).reshape(N, N)
        tipo2D = self.tipo.reshape(N, N)
        inhibMask = (tipo2D == 0)

        # --- MODO 1: CONWAY CLÁSICO DETERMINISTA (B3/S23) ---
        if self.brainMode == 'conway_clasico':
            vecinos = self._contar_vecinos_moore(G)
            nace = (G == 0) & (vecinos == 3)
            sobrevive = (G == 1) & ((vecinos == 2) | (vecinos == 3))
            nxt = (nace | sobrevive | ext2D)
            # Si el freno frontal estimula neuronas inhibitorias, apagan su vecindad
            if self.frontInhib and np.any(ext2D & inhibMask):
                freno = self._contar_vecinos_moore(ext2D & inhibMask) > 0
                nxt = nxt & ~freno
            self.grid = nxt.astype(np.uint8).ravel()
            return

        # --- MODO 2: CONWAY PROBABILÍSTICO + REFRACTARIEDAD (DÍA 1) ---
        if self.brainMode == 'conway_prob':
            vecinos = self._contar_vecinos_moore(G)
            R1 = self.rng.random_matrix((N, N))
            R2 = self.rng.random_matrix((N, N))

            disp = (self.refracGrid == 0)
            nace = (G == 0) & disp & (vecinos == 3) & (R1 < self.pBirth)
            sobrevive = (G == 1) & ((vecinos == 2) | (vecinos == 3)) & (R1 < self.pSurvive)
            espontaneo = (G == 0) & disp & (R2 < self.pSpontaneous)

            nxt = nace | sobrevive | espontaneo | (ext2D & ~inhibMask)

            # Acción de células inhibitorias activas
            inhActivas = (G == 1) & inhibMask | (ext2D & inhibMask)
            if np.any(inhActivas):
                vecInh = self._contar_vecinos_moore(inhActivas)
                nxt = nxt & (vecInh < 2)

            # Actualizar período refractario para celdas que acaban de apagarse
            seApaga = (G == 1) & (~nxt)
            self.refracGrid = np.maximum(0, self.refracGrid - 1)
            if self.refracSteps > 0:
                self.refracGrid[seApaga] = self.refracSteps

            self.grid = nxt.astype(np.uint8).ravel()
            return

        # --- MODO 3: BIO-PLÁSTICO LOCAL (ATP + INHIBICIÓN + HEBB LOCAL - DÍA 2) ---
        if self.brainMode == 'bio_plastico':
            excAct = (G == 1) & (~inhibMask)
            inhAct = ((G == 1) & inhibMask) | (ext2D & inhibMask)

            vecExc = self._contar_vecinos_moore(excAct)
            vecInh = self._contar_vecinos_moore(inhAct)
            balance = vecExc - 1.2 * vecInh

            en2D = self.energia.reshape(N, N)
            factorATP = np.clip(en2D / E_MAX, 0.05, 1.0)
            R = self.rng.random_matrix((N, N))

            # Probabilidad de disparo modulada por umbral local plástico y reserva de ATP
            superaUmbral = (balance >= self.umbralLocal - 0.5) & (balance <= self.umbralLocal + 1.5)
            probAct = self.pBirth * factorATP
            disp = (self.refracGrid == 0) & (en2D > 20.0)

            nxt = (disp & superaUmbral & (R < probAct)) | (disp & (R < self.pSpontaneous))
            nxt = nxt | (ext2D & (~inhibMask))

            # Veto inhibitorio local si hay fuerte inhibición circundante
            if self.veto:
                vetadas = vecInh >= self.vetoK
                self.nVetoed = int(np.count_nonzero(nxt & vetadas))
                nxt = nxt & (~vetadas)

            # Dinámica metabólica (ATP): gasto por activación y recuperación homeostática
            enNueva = np.where(
                nxt,
                en2D - self.atpDecay,
                en2D + self.atpRecovery
            )
            enNueva[ext2D] = np.maximum(enNueva[ext2D], 80.0)
            self.energia = np.clip(enNueva, 0.0, E_MAX).astype(np.float32).ravel()

            # Plasticidad Hebbiana espacial: si una celda se activa junto con sus vecinas,
            # reduce ligeramente su umbral local; si no, relaja hacia el umbral basal (3.0)
            coactivadas = nxt & (vecExc >= 2)
            self.umbralLocal = np.where(
                coactivadas,
                np.maximum(1.6, self.umbralLocal - self.localHebbRate),
                self.umbralLocal + 0.002 * (3.0 - self.umbralLocal)
            ).astype(np.float32)

            seApaga = (G == 1) & (~nxt)
            self.refracGrid = np.maximum(0, self.refracGrid - 1)
            if self.refracSteps > 0:
                self.refracGrid[seApaga] = self.refracSteps

            self.grid = nxt.astype(np.uint8).ravel()

    def decodificar_motores(self):
        """
        [DÍA 1 - TARDE] Decodificación motora (Output):
        Lee la tasa de disparo en las zonas motoras derecha (`maskDerB`) e
        izquierda (`maskIzqB`) y calcula la velocidad lineal (`heldU`) y
        velocidad angular (`heldW`) del vehículo diferencial.
        """
        actv = (self.grid != 0) & (self.tipo == 1)
        actDer = int(np.count_nonzero(actv & self.maskDerB))
        actIzq = int(np.count_nonzero(actv & self.maskIzqB))

        f1 = actDer / self.nDerE if self.nDerE > 0 else 0.0   # Zona derecha -> rueda S1
        f2 = actIzq / self.nIzqE if self.nIzqE > 0 else 0.0   # Zona izquierda -> rueda S2

        wMax, wMin, fMax = 5.0, 1.0, 0.8
        S1 = (f1 * 2) * (wMax - wMin) / (2 * fMax) + wMin if f1 < fMax else wMax
        S2 = (f2 * 2) * (wMax - wMin) / (2 * fMax) + wMin if f2 < fMax else wMax

        self.heldU = (S1 + S2) / 5.0
        self.heldW = (S2 - S1) / 5.0 if self.wiring == 'contra' else (S1 - S2) / 5.0
        self.lastU, self.lastW, self.lastF1, self.lastF2 = self.heldU, self.heldW, f1, f2

    # =================================================================
    # 4. REGISTRO NEUROBIOLÓGICO (RASTER PLOT + FIRING RATE POBLACIONAL)
    # =================================================================
    def _init_neuro_recording(self):
        """Selecciona 100 unidades representativas para el Raster Plot."""
        sens_idx = np.where(self.neuronaEnSensor >= 0)[0]
        der_idx = np.where(self.maskDerB & (self.neuronaEnSensor < 0))[0]
        izq_idx = np.where(self.maskIzqB & (self.neuronaEnSensor < 0))[0]
        other_idx = np.where((~self.maskDerB) & (~self.maskIzqB) & (self.neuronaEnSensor < 0))[0]

        s_pick = sens_idx[:30]
        d_pick = der_idx[:25]
        i_pick = izq_idx[:25]
        o_pick = other_idx[:N_RASTER_NEURONS - (len(s_pick) + len(d_pick) + len(i_pick))]

        self.rasterIndices = np.concatenate([s_pick, d_pick, i_pick, o_pick])
        self.rasterTypes = np.concatenate([
            np.full(len(s_pick), 1, dtype=np.uint8),  # 1: Sensorial (azul/naranja)
            np.full(len(d_pick), 2, dtype=np.uint8),  # 2: Motor Derecho (naranja)
            np.full(len(i_pick), 3, dtype=np.uint8),  # 3: Motor Izquierdo (verde)
            np.full(len(o_pick), 0, dtype=np.uint8),  # 0: Corteza interna (blanco/gris)
        ])
        self.rasterHist.fill(0)
        self.rateGlobalHist.fill(0.0)
        self.rateDerHist.fill(0.0)
        self.rateIzqHist.fill(0.0)

    def _update_neuro_recording(self):
        """Actualiza la ventana temporal del Raster Plot y de la Tasa de Disparo."""
        self.rasterHist[:, :-1] = self.rasterHist[:, 1:]
        self.rasterHist[:, -1] = self.grid[self.rasterIndices]

        self.rateGlobalHist[:-1] = self.rateGlobalHist[1:]
        self.rateDerHist[:-1] = self.rateDerHist[1:]
        self.rateIzqHist[:-1] = self.rateIzqHist[1:]

        self.rateGlobalHist[-1] = float(np.count_nonzero(self.grid)) / float(M)
        self.rateDerHist[-1] = float(self.lastF1)
        self.rateIzqHist[-1] = float(self.lastF2)

    # =================================================================
    # 5. MODELO METABÓLICO 40K DE REFERENCIA Y FÍSICA DEL ENTORNO
    # =================================================================
    def build_adjacency(self):
        idx = np.arange(M)
        ii = idx[:, None]
        jj = idx[None, :]
        if self.sparsity == 'hash':
            a = np.minimum(ii, jj).astype(np.uint64)
            b = np.maximum(ii, jj).astype(np.uint64)
            h = (a * np.uint64(2654435761)) & np.uint64(MASK32)
            h = (h ^ ((((b ^ (b >> np.uint64(13))) * np.uint64(40503))) & np.uint64(MASK32))) & np.uint64(MASK32)
            h = (((h ^ (h >> np.uint64(13))) * np.uint64(3266489917))) & np.uint64(MASK32)
            h = (h ^ (h >> np.uint64(16))) & np.uint64(MASK32)
            keep = (h % np.uint64(ADJ_MOD)) == 0
        else:
            keep = ((ii + jj) % ADJ_MOD) == 0

        dx = PXc[ii].astype(np.float64) - PXc[jj].astype(np.float64)
        dy = PYc[ii].astype(np.float64) - PYc[jj].astype(np.float64)
        d2 = dx * dx + dy * dy
        keep &= (d2 <= float(self.ADJ_DIST) * float(self.ADJ_DIST))
        np.fill_diagonal(keep, False)

        src, dst = np.nonzero(keep)
        self.src = src.astype(np.int64)
        self.dst = dst.astype(np.int64)
        self.adjDist = np.sqrt(d2[src, dst]).astype(np.float32)
        self.deg = np.bincount(self.src, minlength=M).astype(np.float64)
        self.w = np.full(self.src.size, W_INIT, np.float32)
        self._inR_edge = self.adjDist <= np.float32(self.inhRadius)

    def _refresh_edge_masks(self):
        self._inR_edge = self.adjDist <= np.float32(self.inhRadius)
        self._excJ = self.tipo[self.dst] == 1
        self._inhJ_inR = (~self._excJ) & self._inR_edge

    def _step_metabolico_40k(self, t, extStim):
        src, dst, w = self.src, self.dst, self.w
        aj = self.grid[dst] != 0
        mE = aj & self._excJ
        mI = aj & self._inhJ_inR

        V = np.bincount(src[mE | mI], minlength=M).astype(np.float64)
        E = np.bincount(src[mE], weights=w[mE].astype(np.float64), minlength=M)
        I = np.bincount(src[mI], weights=w[mI].astype(np.float64), minlength=M)
        Ic = np.bincount(src[mI], minlength=M).astype(np.float64)

        net = E - I
        if self.hebb:
            net = net + np.where(self.sens > 0, self.sensBonus, 0.0)
        activar = net >= ACT_THRESH

        ext = extStim != 0
        gridB = self.grid != 0

        if self.pureCA:
            act = activar | ext
            if self.veto:
                vetoed = Ic >= self.vetoK
                self.nVetoed = int(np.count_nonzero(act & vetoed))
                act = act & ~vetoed
            else:
                self.nVetoed = 0
            self.sens = np.where(
                act & (V > 0), self.sensWin,
                np.where(self.sens > 0, self.sens.astype(np.int32) - 1, 0)
            ).astype(np.uint8)
            self.grid = act.astype(np.uint8)
            return

        e0 = self.energia.astype(np.float64)
        costo = self.costo.astype(np.float64)
        consumo = ((V - Ic) + self.inhCost * Ic) * np.abs(costo)

        en = e0.copy()
        m = activar & (e0 > 0)
        en[m] = e0[m] - consumo[m] * 0.05
        m = activar & (en < 0) & (en > -20)
        en[m] = np.abs(en[m]) * 0.5

        sep = np.where(activar & (en > -20), 0.0, en)
        gu = self.guardada.astype(np.float64)
        m = sep < 0
        gu[m] = gu[m] + np.abs(sep[m]) * 0.01

        cons10 = consumo * 10.0
        m = (V >= NEIGH_HIGH) & (gu >= cons10)
        en[m] = en[m] + gu[m]
        gu[gu >= cons10] = 0.0

        aRaw = activar & (e0 > 0)
        if t > 0:
            aRaw = aRaw | ((e0 > 0) & (self.facilitado > 0) & (~activar) & (V > 0))

        en[V >= NEIGH_HIGH] += 50.0

        fac = self.facilitado.astype(np.int32)
        fac[(e0 > 0) & (consumo > 0) & gridB & (~aRaw)] -= 1
        fac[(e0 > 0) & activar] = 2
        np.clip(fac, 0, 3, out=fac)

        c = costo + np.where(self.tipo == 1, COST_STEP, -COST_STEP) * self.costMult

        act = aRaw.copy()
        act[ext] = True
        m = ext & (en < 10.0)
        en[m] += E_SENSE_BOOST

        if self.veto:
            vetoed = Ic >= self.vetoK
            self.nVetoed = int(np.count_nonzero(act & vetoed))
            act = act & ~vetoed
        else:
            self.nVetoed = 0

        np.clip(en, 0.0, E_MAX, out=en)

        self.sens = np.where(
            act & (V > 0), self.sensWin,
            np.where(self.sens > 0, self.sens.astype(np.int32) - 1, 0)
        ).astype(np.uint8)
        nxt = act.astype(np.uint8)

        if self.hetero:
            self._apply_plasticity(nxt)

        self.grid = nxt
        self.energia = en.astype(np.float32)
        self.guardada = gu.astype(np.float32)
        self.facilitado = fac.astype(np.uint8)
        self.costo = c.astype(np.float32)

    def _apply_plasticity(self, nxt):
        m = (nxt[self.src] != 0) & (self.grid[self.dst] != 0)
        wf = self.w.astype(np.float64)
        wf[m] = wf[m] + HEBB_LR * self.hetSev * wf[m]
        self.w = wf.astype(np.float32)
        sums = np.bincount(self.src, weights=self.w.astype(np.float64), minlength=M)
        scale = np.ones(M, np.float64)
        ok = (sums > 0) & (self.deg > 0)
        scale[ok] = self.deg[ok] / sums[ok]
        self.w = (self.w.astype(np.float64) * scale[self.src]).astype(np.float32)

    # ----------------------------------------------------------------
    # Construcción de Arena (Cuadrada y Circular)
    # ----------------------------------------------------------------
    def _paint(self, x, y, t):
        x = js_round(x)
        y = js_round(y)
        if 0 <= x < PLANE and 0 <= y < PLANE and not self.obstacle[y, x]:
            self.obstacle[y, x] = t
            self._obs_list.append((x, y))

    def _diag12(self, x, y):
        for i in range(12):
            self._paint(x + i, y + i, 2)

    def _square12(self, x, y):
        for d in range(4):
            self._paint(x + d, y, 4)
            self._paint(x + d, y + 3, 4)
        self._paint(x, y + 1, 4)
        self._paint(x, y + 2, 4)
        self._paint(x + 3, y + 1, 4)
        self._paint(x + 3, y + 2, 4)

    def _cross12(self, cx, cy):
        for d in range(-3, 4):
            self._paint(cx, cy + d, 1)
        for d in (-1, -2, -3, 1, 2):
            self._paint(cx + d, cy, 1)

    def _place_fig(self, kind, cx, cy):
        if kind == 'diag':
            self._diag12(cx, cy)
        elif kind == 'cuad':
            self._square12(cx, cy)
        elif kind == 'cruz':
            self._cross12(cx + 3, cy + 3)
        else:
            self._paint(cx, cy, 3)

    def _build_multi(self, kinds):
        k = 0
        cy = 50
        while cy < PLANE - 12:
            cx = 50
            while cx < PLANE - 12:
                self._place_fig(kinds[k % len(kinds)], cx, cy)
                k += 1
                cx += self.objSEP
            cy += self.objSEP

    def _banana_blob(self, cx, cy):
        cells = []
        for dy in range(3):
            for dx in range(3):
                self._paint(cx + dx, cy + dy, 5)
                cells.append((cx + dx, cy + dy))
        self.bananas.append({'cx': cx + 1, 'cy': cy + 1, 'cells': cells, 'done': False})

    def _circle_mask(self, center, radius):
        yy, xx = np.ogrid[:PLANE, :PLANE]
        return ((xx.astype(np.float64) - center[0]) ** 2 +
                (yy.astype(np.float64) - center[1]) ** 2 <= radius ** 2)

    def build_field(self):
        self.obstacle.fill(0)
        self._obs_list = []
        self.bananas = []
        self.bananasTotal = 0
        self.bananasCollected = 0

        if self.arenaShape == 'circular':
            for idx, center in enumerate(OBSTACULOS_CIRCULARES):
                mask = self._circle_mask(center, RADIO_OBSTACULO)
                self.obstacle[mask] = (idx % 4) + 1
            ys, xs = np.nonzero(self.obstacle)
            self.obs_x = xs.astype(np.int32)
            self.obs_y = ys.astype(np.int32)
            self._obs_list = list(zip(self.obs_x.tolist(), self.obs_y.tolist()))
            self._obs_dirty = True
            self._rebuild_dilated()
            return

        f = self.fieldType
        if f == 'mixto':
            self._build_multi(['cruz', 'diag', 'punto'])
        elif f == 'todas':
            self._build_multi(['cruz', 'diag', 'cuad', 'punto'])
        elif f == 'cuaddiag':
            self._build_multi(['cuad', 'diag'])
        elif f == 'bananas':
            cy = 50
            while cy < PLANE - 12:
                cx = 50
                while cx < PLANE - 12:
                    self._banana_blob(cx, cy)
                    cx += self.objSEP
                cy += self.objSEP
            self.bananasTotal = len(self.bananas)
        else:
            self._build_multi([f])

        if self._obs_list:
            arr = np.array(self._obs_list, np.int32)
            self.obs_x, self.obs_y = arr[:, 0], arr[:, 1]
        else:
            self.obs_x = np.zeros(0, np.int32)
            self.obs_y = np.zeros(0, np.int32)
        self._obs_dirty = True
        self._rebuild_dilated()

    def _rebuild_dilated(self):
        occ = self.obstacle != 0
        r = int(self.senseR)
        if r <= 0:
            self._dilated = occ
            return
        dil = np.zeros_like(occ)
        for b in range(-r, r + 1):
            ys = slice(max(0, -b), PLANE - max(0, b))
            yd = slice(max(0, b), PLANE - max(0, -b))
            for a in range(-r, r + 1):
                xs = slice(max(0, -a), PLANE - max(0, a))
                xd = slice(max(0, a), PLANE - max(0, -a))
                dil[yd, xd] |= occ[ys, xs]
        self._dilated = dil

    def check_bananas(self):
        if self.arenaShape != 'cuadrada' or self.fieldType != 'bananas' or not self.bananasTotal:
            return
        R2 = (CAR_R + 6) ** 2
        for b in self.bananas:
            if b['done']:
                continue
            dx = self.carX - b['cx']
            dy = self.carY - b['cy']
            if dx * dx + dy * dy <= R2:
                b['done'] = True
                self.bananasCollected += 1
                for (x, y) in b['cells']:
                    self.obstacle[y, x] = 6
                self._obs_dirty = True

    @staticmethod
    def _ray_circle_distance(origin, direction, center, radius):
        ox, oy = origin
        dx, dy = direction
        cx, cy = center
        fx, fy = ox - cx, oy - cy
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return math.inf
        root = math.sqrt(max(0.0, disc))
        t1 = (-b - root) / 2.0
        t2 = (-b + root) / 2.0
        candidates = [t for t in (t1, t2) if t >= 0.0]
        return min(candidates) if candidates else math.inf

    def ray_dist(self, ox, oy, ang):
        if self.arenaShape == 'circular':
            origin = np.array([ox, oy], dtype=np.float64)
            direction = np.array([math.cos(ang), math.sin(ang)], dtype=np.float64)
            distances = []
            bd = self._ray_circle_distance(origin, direction, CENTRO_ARENA, RADIO_ARENA - CAR_R)
            if np.isfinite(bd):
                distances.append(bd)
            for center in OBSTACULOS_CIRCULARES:
                od = self._ray_circle_distance(origin, direction, center, RADIO_OBSTACULO + CAR_R)
                if np.isfinite(od):
                    distances.append(od)
            if not distances:
                return float(MAXR)
            return float(min(MAXR, max(0.0, min(distances))))

        d = self._ds
        x = ox + math.cos(ang) * d
        y = oy + math.sin(ang) * d
        oob = (x < 0) | (y < 0) | (x >= PLANE) | (y >= PLANE)
        xi = np.floor(x + 0.5).astype(np.int64)
        yi = np.floor(y + 0.5).astype(np.int64)
        np.clip(xi, 0, PLANE - 1, out=xi)
        np.clip(yi, 0, PLANE - 1, out=yi)
        stop = oob | (self._dilated[yi, xi] & ~oob)
        k = int(np.argmax(stop))
        if stop[k]:
            return float(d[k])
        return float(MAXR)

    def _O(self, x, y):
        if 0 <= x < PLANE and 0 <= y < PLANE:
            return self.obstacle[y, x]
        return 0

    def hits_obstacle(self, x, y):
        if self.arenaShape == 'circular':
            point = np.array([x, y], dtype=np.float64)
            if np.linalg.norm(point - CENTRO_ARENA) + CAR_R >= RADIO_ARENA:
                return 2
            if np.any(np.linalg.norm(OBSTACULOS_CIRCULARES - point, axis=1) <= RADIO_OBSTACULO + CAR_R):
                return 1
            return 0

        if x < CAR_R or y < CAR_R or x >= PLANE - CAR_R or y >= PLANE - CAR_R:
            return 2
        for a in range(6):
            an = a * math.pi / 3.0
            if self._O(js_round(x + math.cos(an) * CAR_R), js_round(y + math.sin(an) * CAR_R)):
                return 1
        return 1 if self._O(js_round(x), js_round(y)) else 0

    def choose_sensor_neurons(self):
        self.chosen = np.zeros(M, np.uint8)
        all_ = [i for i in range(M) if self.neuronaEnSensor[i] >= 0 and not self.sensorInhib[i]]
        rng = self.rng
        for i in range(len(all_) - 1, 0, -1):
            j = int(rng() * (i + 1))
            all_[i], all_[j] = all_[j], all_[i]
        k = min(self.Kchosen, len(all_))
        for i in range(k):
            self.chosen[all_[i]] = 1

    def reset_car(self):
        self.carX = float(CENTRO_ARENA[0]) if self.arenaShape == 'circular' else PLANE * 0.5
        self.carY = float(CENTRO_ARENA[1]) if self.arenaShape == 'circular' else PLANE * 0.5
        self.carDir = self.rng() * 2 * math.pi
        self.crashesConsec = 0
        self.trail = []

    def move_car(self):
        dt = 1.0 / 5.0
        self.carDir = math.fmod(self.carDir - 5.0 * self.heldW * dt, 2 * math.pi)
        adv = self.heldU * dt * self.carSpeed
        nx = self.carX + math.cos(self.carDir) * adv
        ny = self.carY + math.sin(self.carDir) * adv
        hit = self.hits_obstacle(nx, ny)
        if hit:
            if hit == 2:
                self.crashBorder += 1
            else:
                self.crashObj += 1
            self.flash = 6
            self.flashColor = '#f85149' if hit == 2 else '#ffd33d'
            self.crashesConsec += 1
            self.carDir = math.fmod(self.carDir + math.pi / 2 + (self.rng() - 0.5), 2 * math.pi)
            if self.crashesConsec >= CRASH_RESET:
                self.reset_car()
        else:
            self.carX, self.carY = nx, ny
            self.crashesConsec = max(0, self.crashesConsec - 1)
            self.distTraveled += adv
            self.trail.append((self.carX, self.carY))
            if len(self.trail) > 240:
                self.trail.pop(0)
        self.check_bananas()

    def controller_step(self):
        for a in range(NRAY):
            self.lastDist[a] = self.ray_dist(self.carX, self.carY, self.carDir + SENSOR_ANGLES[a])
        L = self.lastDist
        pF = max(0.0, 1 - L[0] / MAXR)
        pR = max(0.0, (3 - (L[1] + L[2] + L[3]) / MAXR) / 3)
        pL = max(0.0, (3 - (L[4] + L[5] + L[6]) / MAXR) / 3)
        baseU, TURN = 1.2, 1.8
        if self.controller == 'levy':
            near = max(pF, pR, pL)
            if near > 0.45:
                self.heldU = baseU * 0.6
                self.heldW = TURN * ((pR - pL) + (0.4 if pR + pL < 0.1 else 0.0))
                self.levyRem = 0
            else:
                if self.levyRem <= 0:
                    u = max(1e-4, self.rng())
                    self.levyRem = min(300, js_round(u ** (-1 / 1.5)))
                    self.levyTurn = (self.rng() - 0.5) * 0.6
                self.heldU = baseU
                self.heldW = self.levyTurn
                self.levyRem -= 1
        else:
            hw = TURN * (pR - pL)
            if self.controller == 'braitSeek':
                hw = -hw
            if self.controller == 'braitOsc':
                self.oscP += 0.15
                hw += 0.5 * math.sin(self.oscP)
            self.heldU = max(0.2, baseU * (1 - 0.8 * pF))
            self.heldW = hw
        self.lastU, self.lastW = self.heldU, self.heldW
        self.lastExtStim = np.zeros(M, np.uint8)

    def registrar_tramo(self):
        c = self.crashObj + self.crashBorder
        self.tramos.append((self.step, (c - self.tramoPrev) / TRAMO * 1000.0))
        self.tramoPrev = c

    def texto_tramos(self):
        if not self.tramos:
            return 'tramos de %d pasos — todavía sin datos' % TRAMO
        cab = ''.join(pad('%gk' % (h / 1000.0), 7) for h, _ in self.tramos)
        val = ''.join(pad(tofixed(v, 1), 7) for _, v in self.tramos)
        n = len(self.tramos)
        ult = sum(v for _, v in self.tramos[-3:]) / min(3, n)
        s = ('choques/1000, tramos de %d pasos\n' % TRAMO +
             'hasta ' + cab + '\ntasa  ' + val +
             '\n\nprimer tramo ' + tofixed(self.tramos[0][1], 1) +
             '   últimos 3 (meseta) ' + tofixed(ult, 1))
        if n >= 8:
            s += '   -> ' + ('ADAPTA' if ult < self.tramos[0][1] * 0.7 else 'sin adaptación clara')
        else:
            s += '   (seguir hasta 40k+)'
        return s

    def maybe_warm(self):
        if (self.fieldType == 'bananas' and self.bananaArmed and
                self.bananaWarm > 0 and self.step >= self.bananaWarm):
            self.bananaArmed = False
            self.build_field()
            self.bananasCollected = 0
            self.distTraveled = 0.0
            self.crashObj = 0
            self.crashBorder = 0
            self.measStart = self.step

    def advance(self):
        self.maybe_warm()
        if self.controller != 'automata':
            self.controller_step()
            self.move_car()
            self._update_neuro_recording()
            self.step += 1
            if self.step % TRAMO == 0:
                self.registrar_tramo()
            return

        extStim = self.codificar_sensores()
        self.lastExtStim = extStim
        self.paso_automata_local(self.step, extStim)
        self.moveCounter += 1
        if self.moveCounter >= self.moveEvery:
            self.decodificar_motores()
            self.moveCounter = 0
        self.move_car()
        self._update_neuro_recording()
        self.step += 1
        if self.step % TRAMO == 0:
            self.registrar_tramo()

    def reset(self):
        self.rng = Mulberry32(self.seedValue)
        self.build_field()
        self.definir_zonas_sensoriomotrices()
        self.choose_sensor_neurons()
        self.init_condiciones_locales()
        self.reset_car()
        self.crashObj = 0
        self.crashBorder = 0
        self.distTraveled = 0.0
        self.step = 0
        self.moveCounter = 0
        self.lastExtStim = np.zeros(M, np.uint8)
        self.tramos = []
        self.tramoPrev = 0
        self.bananaArmed = (self.bananaWarm > 0)
        self.measStart = 0
        self.oscP = 0.0
        self.levyRem = 0
        self.flash = 0
        self.decodificar_motores()

    def hud_text(self):
        act = int(np.count_nonzero(self.grid))
        sumE = float(self.energia.sum())
        dstr = ' '.join(pad(tofixed(d, 0), 3) for d in self.lastDist)
        s = ('paso %s | modo=%s | arena=%s | carro (%s, %s) dir %s°\n'
             % (pad(self.step, 5), self.brainMode, self.arenaShape,
                tofixed(self.carX, 0), tofixed(self.carY, 0),
                tofixed(self.carDir * 180 / math.pi, 0)))
        s += ('choques obj=%s | choques borde=%s | distancia=%s (seguidos %d)\n'
              % (pad(self.crashObj, 4), pad(self.crashBorder, 4),
                 tofixed(self.distTraveled, 0), self.crashesConsec))
        s += ('ruedas S1(der) f1=%s | S2(izq) f2=%s  ->  u=%s  w=%s  [%s]\n'
              % (tofixed(self.lastF1, 2), tofixed(self.lastF2, 2),
                 tofixed(self.lastU, 2), tofixed(self.lastW, 2),
                 'ipsilateral/evita' if self.wiring == 'ipsi' else 'contralateral/embiste'))
        s += ('sensores [0 +30 +45 +90 -30 -45 -90] = %s | activas=%d/%d (%.1f%%) | ATP medio=%s'
              % (dstr, act, M, 100.0 * act / M, tofixed(sumE / M, 0)))
        return s


# Alias de compatibilidad con scripts externos
Sim = BootcampSim


# =====================================================================
# 6. INTERFAZ GRÁFICA (ARENA + GRILLA + PANEL NEUROCIENCIAS EN VIVO)
# =====================================================================
BG = '#0d1117'
PANEL = '#161b22'
LINE = '#30363d'
TXT = '#e6edf3'
MUTED = '#8b949e'
ACCENT = '#58a6ff'
EXC = '#39d353'
INH = '#f85149'

AW = 420          # Canvas Arena
BW = 400          # Canvas Cerebro 40x40 (10 px por celda)
NW_W = 360        # Ancho Canvas Neurociencias (Raster + Firing Rate)
AS = AW / PLANE
CS = BW // N


def _rgb(h):
    return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))


OBS_COLORS = {
    1: _rgb('#f0f6fc'), 2: _rgb('#ffa657'), 3: _rgb('#f778ba'),
    4: _rgb('#56d4dd'), 5: _rgb('#ffd21f'), 6: _rgb('#ffffff')
}


class App:
    def __init__(self, root, sim):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.sim = sim

        self.running = False
        self.fps = 20
        self.ff_thread = None
        self.ff_total = 0
        self.ff_left = 0
        self.ff_lock = threading.Lock()

        root.title('Bootcamp IA Bioinspirada — Caja de Arena Conway + Vehículo + Neuro-Visualizaciones')
        root.configure(bg=BG)

        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure('.', background=PANEL, foreground=TXT, fieldbackground='#21262d')
        style.configure('TCombobox', fieldbackground='#21262d', background='#21262d')

        outer = tk.Frame(root, bg=BG)
        outer.pack(fill='both', expand=True, padx=10, pady=6)

        tk.Label(
            outer, bg=BG, fg=TXT, font=('Helvetica', 14, 'bold'), anchor='w',
            text='Bootcamp de IA Bioinspirada: "Aprendizaje desde propiedades locales" (Conway + Braitenberg)'
        ).pack(fill='x')

        top = tk.Frame(outer, bg=BG)
        top.pack(fill='x', pady=(4, 0))

        # 1. Panel Arena Física
        pa = tk.Frame(top, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        pa.pack(side='left', padx=(0, 8), anchor='n')
        tk.Label(pa, text='1. ARENA DE NAVEGACIÓN', bg=PANEL, fg=MUTED,
                 font=('Helvetica', 9, 'bold')).pack(anchor='w', padx=8, pady=(5, 2))
        self.cvA = tk.Canvas(pa, width=AW, height=AW, bg='#000', highlightthickness=0)
        self.cvA.pack(padx=8, pady=(0, 8))

        # 2. Panel Grilla del Autómata
        pb = tk.Frame(top, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        pb.pack(side='left', padx=(0, 8), anchor='n')
        tk.Label(pb, text='2. PROTO-CEREBRO 40×40 (CONWAY / BIO)', bg=PANEL, fg=MUTED,
                 font=('Helvetica', 9, 'bold')).pack(anchor='w', padx=8, pady=(5, 2))
        self.cvB = tk.Canvas(pb, width=BW, height=BW, bg='#000', highlightthickness=0)
        self.cvB.pack(padx=8, pady=(0, 8))

        # 3. Panel Visualizaciones Neurocientíficas (Raster Plot + Tasa de Disparo)
        pn = tk.Frame(top, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        pn.pack(side='left', anchor='n')
        tk.Label(pn, text='3. ANÁLISIS NEUROBIOLÓGICO EN VIVO', bg=PANEL, fg=MUTED,
                 font=('Helvetica', 9, 'bold')).pack(anchor='w', padx=8, pady=(5, 2))
        tk.Label(pn, text='A) Raster Plot (azul=sens, nar=mot der, ver=mot izq)',
                 bg=PANEL, fg=MUTED, font=('Helvetica', 8)).pack(anchor='w', padx=8)
        self.cvRaster = tk.Canvas(pn, width=NW_W, height=200, bg='#05080c', highlightthickness=0)
        self.cvRaster.pack(padx=8, pady=(2, 6))

        tk.Label(pn, text='B) Tasa de Disparo Poblacional / LFP (der, izq, global)',
                 bg=PANEL, fg=MUTED, font=('Helvetica', 8)).pack(anchor='w', padx=8)
        self.cvRate = tk.Canvas(pn, width=NW_W, height=170, bg='#05080c', highlightthickness=0)
        self.cvRate.pack(padx=8, pady=(2, 8))

        # Panel inferior de métricas HUD + Tramos
        info_frame = tk.Frame(outer, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        info_frame.pack(fill='x', pady=(6, 0))
        self.hud = tk.Label(info_frame, bg=PANEL, fg=TXT, font=('Menlo', 9),
                            justify='left', anchor='w', text='')
        self.hud.pack(side='left', padx=8, pady=4)
        self.tramosLbl = tk.Label(info_frame, bg=PANEL, fg=ACCENT, font=('Menlo', 9),
                                  justify='left', anchor='w', text='')
        self.tramosLbl.pack(side='right', padx=12, pady=4)

        self._build_controls(outer)

        self.imgA = None
        self.imgB = None
        self.imgRaster = None
        self.itemA = self.cvA.create_image(0, 0, anchor='nw')
        self.itemB = self.cvB.create_image(0, 0, anchor='nw')
        self.itemRaster = self.cvRaster.create_image(0, 0, anchor='nw')
        self._arena_bg = None
        self._brain_static_dirty = True

        self.draw()
        self.update_hud()
        self.root.after(50, self._tick)

    def _row(self, parent):
        f = self.tk.Frame(parent, bg=PANEL)
        f.pack(fill='x', pady=2)
        return f

    def _btn(self, parent, text, cmd, kind='sec'):
        tk = self.tk
        bg = ACCENT if kind == 'pri' else '#21262d'
        fg = '#0d1117' if kind == 'pri' else TXT
        font = ('Helvetica', 9, 'bold') if kind == 'pri' else ('Helvetica', 9)
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                      activebackground=bg, relief='flat', font=font,
                      padx=8, pady=2, highlightthickness=0)
        b.pack(side='left', padx=3)
        return b

    def _toggle(self, parent, label, getter, setter, on_color=EXC):
        tk = self.tk
        b = tk.Button(parent, relief='flat', font=('Helvetica', 9), padx=8, pady=2,
                      highlightthickness=0)

        def paint():
            on = getter()
            b.config(text='%s: %s' % (label, 'ON' if on else 'OFF'),
                     bg=on_color if on else '#21262d',
                     fg='#0d1117' if on else TXT)

        def click():
            setter(not getter())
            paint()
        b.config(command=click)
        paint()
        b.pack(side='left', padx=3)
        return b

    def _slider(self, parent, label, lo, hi, res, val, cb, width=95, fmt=None):
        tk = self.tk
        box = tk.Frame(parent, bg=PANEL)
        box.pack(side='left', padx=5)
        tk.Label(box, text=label, bg=PANEL, fg=MUTED, font=('Helvetica', 9)).pack(side='left')
        var = tk.DoubleVar(value=val)
        vl = tk.Label(box, text=(fmt(val) if fmt else self._fmtnum(val)), bg=PANEL, fg=TXT,
                      font=('Menlo', 9), width=5, anchor='w')

        def on_change(_v):
            v = var.get()
            vl.config(text=(fmt(v) if fmt else self._fmtnum(v)))
            cb(v)
        s = tk.Scale(box, from_=lo, to=hi, resolution=res, orient='horizontal', variable=var,
                     command=on_change, showvalue=0, length=width, bg=PANEL, fg=TXT,
                     troughcolor='#21262d', highlightthickness=0, sliderrelief='flat',
                     activebackground=ACCENT, bd=0)
        s.pack(side='left')
        vl.pack(side='left', padx=(2, 0))
        return var

    @staticmethod
    def _fmtnum(v):
        return str(int(v)) if float(v) == int(v) else ('%.2f' % v)

    def _combo(self, parent, label, options, value, cb, width=22):
        tk, ttk = self.tk, self.ttk
        box = tk.Frame(parent, bg=PANEL)
        box.pack(side='left', padx=5)
        tk.Label(box, text=label, bg=PANEL, fg=MUTED, font=('Helvetica', 9)).pack(side='left')
        labels = [l for _, l in options]
        keys = [k for k, _ in options]
        var = tk.StringVar(value=dict(options)[value])
        c = ttk.Combobox(box, values=labels, textvariable=var, state='readonly', width=width)
        c.pack(side='left', padx=3)
        c.bind('<<ComboboxSelected>>', lambda e: cb(keys[labels.index(var.get())]))
        return var

    def _build_controls(self, parent):
        tk = self.tk
        s = self.sim
        panel = tk.Frame(parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        panel.pack(fill='x', pady=(6, 0), ipadx=4, ipady=3)

        # Fila 1: Ejecución y Modo de Autómata
        r = self._row(panel)
        self.playBtn = self._btn(r, '▶ Iniciar', self.toggle_run, 'pri')
        self._btn(r, 'Paso', lambda: (not self.running) and (self.sim.advance(), self.draw(), self.update_hud()))
        self._btn(r, '↺ Reiniciar', self.do_reset)
        for n, lbl in ((5000, '⏩ +5k'), (20000, '⏩ +20k'), (40000, '⏩ +40k')):
            self._btn(r, lbl, lambda n=n: self.fast_forward(n))
        self.ffstatus = tk.Label(r, text='paso 0', bg=PANEL, fg=MUTED, font=('Helvetica', 9))
        self.ffstatus.pack(side='left', padx=6)

        self._combo(
            r, 'Regla Autómata',
            [
                ('conway_clasico', 'Día 1A: Conway Clásico (B3/S23)'),
                ('conway_prob', 'Día 1B: Conway Probabilístico'),
                ('bio_plastico', 'Día 2: Bio-Plástico (ATP + Hebb)'),
                ('metabolico_40k', 'Ref: Cerebro Metabólico 40k'),
            ],
            s.brainMode, self._set_brainmode, width=28
        )

        self._combo(
            r, 'Geometría Arena',
            [('cuadrada', 'Cuadrada (1000×1000)'), ('circular', 'Circular (6 obstáculos)')],
            s.arenaShape, self._set_arenashape, width=20
        )

        # Fila 2: Parámetros de Reglas Locales (Día 1 y Día 2)
        r = self._row(panel)
        self._slider(r, 'P(Nacer)', 0.1, 1.0, 0.05, s.pBirth, lambda v: setattr(s, 'pBirth', float(v)))
        self._slider(r, 'P(Sobrevivir)', 0.1, 1.0, 0.05, s.pSurvive, lambda v: setattr(s, 'pSurvive', float(v)))
        self._slider(r, 'Ruido basal', 0.0, 0.02, 0.001, s.pSpontaneous,
                     lambda v: setattr(s, 'pSpontaneous', float(v)), fmt=lambda v: '%.3f' % v)
        self._slider(r, 'Refractario', 0, 5, 1, s.refracSteps, lambda v: setattr(s, 'refracSteps', int(v)))
        self._slider(r, 'Gasto ATP', 0, 80, 5, s.atpDecay, lambda v: setattr(s, 'atpDecay', float(v)))
        self._slider(r, 'Semillas %', 5, 90, 5, int(s.seedFrac * 100), self._set_seedfrac,
                     fmt=lambda v: '%d%%' % int(v))

        # Fila 3: Sensorio-motriz y Entorno
        r = self._row(panel)
        self._slider(r, 'FPS', 1, 60, 1, 20, lambda v: setattr(self, 'fps', max(1, int(v))))
        self._slider(r, 'Vel. Carro', 2, 40, 1, 14, lambda v: setattr(s, 'carSpeed', float(v)))
        self._slider(r, 'Ganancia Sens.', 0.5, 4, 0.25, 3, lambda v: setattr(s, 'sensGain', float(v)))
        self._toggle(r, 'Freno Frontal', lambda: s.frontInhib, lambda v: setattr(s, 'frontInhib', v))
        self._toggle(r, 'Input Sensorial', lambda: s.sensFeedback, lambda v: setattr(s, 'sensFeedback', v))
        self._combo(r, 'Cableado', [('ipsi', 'Ipsilateral (evita)'), ('contra', 'Contralateral (embiste)')],
                    s.wiring, lambda k: setattr(s, 'wiring', k), width=18)
        self._combo(r, 'Campo', [('mixto', 'Mixto'), ('todas', 'Todas'), ('bananas', '🍌 Bananas')],
                    s.fieldType, self._set_field, width=12)

    def _busy(self):
        return self.ff_thread is not None and self.ff_thread.is_alive()

    def stop_run(self):
        self.running = False
        self.playBtn.config(text='▶ Iniciar')

    def toggle_run(self):
        if self._busy():
            return
        self.running = not self.running
        self.playBtn.config(text='⏸ Pausar' if self.running else '▶ Iniciar')

    def do_reset(self, rebuild_adj=False):
        if self._busy():
            return
        self.stop_run()
        if rebuild_adj:
            self.sim.build_adjacency()
        self.sim.reset()
        self._brain_static_dirty = True
        self.ffstatus.config(text='paso 0')
        self.draw()
        self.update_hud()

    def _set_brainmode(self, mode):
        self.sim.brainMode = mode
        self.do_reset()

    def _set_arenashape(self, shape):
        self.sim.arenaShape = shape
        self.do_reset()

    def _set_field(self, k):
        self.sim.fieldType = k
        self.do_reset()

    def _set_seedfrac(self, v):
        self.sim.seedFrac = float(v) / 100.0
        self.do_reset()

    def fast_forward(self, n):
        if self._busy():
            return
        self.stop_run()
        self.ff_total = n
        self.ff_left = n
        self.ffstatus.config(text='avanzando… 0%')

        def work():
            left = n
            while left > 0:
                k = min(500, left)
                for _ in range(k):
                    self.sim.advance()
                left -= k
                with self.ff_lock:
                    self.ff_left = left
        self.ff_thread = threading.Thread(target=work, daemon=True)
        self.ff_thread.start()
        self.root.after(120, self._poll_ff)

    def _poll_ff(self):
        with self.ff_lock:
            left = self.ff_left
        if self._busy():
            pct = int(round((self.ff_total - left) / self.ff_total * 100))
            self.ffstatus.config(text='avanzando… %d%% (%d)' % (pct, self.sim.step))
            self.root.after(120, self._poll_ff)
        else:
            self.ffstatus.config(text='paso %d' % self.sim.step)
            self.draw()
            self.update_hud()

    def _tick(self):
        if self.running and not self._busy():
            self.sim.advance()
            self.draw()
            self.update_hud()
        self.root.after(max(1, int(1000 / self.fps)), self._tick)

    def _photo(self, arr):
        h, w = arr.shape[:2]
        data = b'P6 %d %d 255 ' % (w, h) + np.ascontiguousarray(arr, np.uint8).tobytes()
        return self.tk.PhotoImage(data=data, format='PPM')

    def _arena_image(self):
        s = self.sim
        img = np.zeros((AW, AW, 3), np.uint8)
        if s.obs_x.size:
            types = s.obstacle[s.obs_y, s.obs_x]
            cols = np.zeros((s.obs_x.size, 3), np.uint8)
            for t, c in OBS_COLORS.items():
                cols[types == t] = c
            px = np.clip((s.obs_x * AS).astype(np.int32), 0, AW - 1)
            py = np.clip((s.obs_y * AS).astype(np.int32), 0, AW - 1)
            img[py, px] = cols
        return img

    def _brain_static(self):
        s = self.sim
        base = np.zeros((M, 3), np.float64)
        base[s.maskIzqB] = np.array((63, 185, 80)) * 0.18
        base[s.maskDerB] = np.array((219, 109, 40)) * 0.18
        ell = s.neuronaEnSensor >= 0
        lat = ell & (s.sensorInhib == 0)
        inh = ell & (s.sensorInhib != 0)
        base[lat] = base[lat] * 0.45 + np.array((31, 111, 235)) * 0.55
        base[inh] = base[inh] * 0.55 + np.array((248, 81, 73)) * 0.45
        self._bstatic = base.reshape(N, N, 3)

    def _brain_image(self):
        s = self.sim
        if self._brain_static_dirty:
            self._brain_static()
            self._brain_static_dirty = False
        base = self._bstatic
        img = np.repeat(np.repeat(base.astype(np.uint8), CS, 0), CS, 1)
        view = img.reshape(N, CS, N, CS, 3).transpose(0, 2, 1, 3, 4)

        act = (s.grid != 0).reshape(N, N)
        alpha = (0.4 + 0.6 * np.clip(s.energia.astype(np.float64) / E_MAX, 0, 1)).reshape(N, N, 1)
        col = np.where((s.tipo == 1).reshape(N, N, 1), np.array(_rgb(EXC)), np.array(_rgb(INH))).astype(np.float64)
        comp = base * (1 - alpha) + col * alpha
        stim = (s.lastExtStim != 0).reshape(N, N)
        comp = np.where(stim[:, :, None], np.array(_rgb('#ffa657'), dtype=np.float64), comp)
        paint = act | stim
        inner = view[:, :, 1:, 1:, :]
        view[:, :, 1:, 1:, :] = np.where(
            paint[:, :, None, None, None],
            comp[:, :, None, None, :].astype(np.uint8),
            inner
        )
        return img

    def _raster_image(self):
        """Genera la imagen RGB del Raster Plot (100 neuronas x 220 pasos, escalada a 200x360)."""
        s = self.sim
        H_raw, W_raw = N_RASTER_NEURONS, HIST_LEN
        canvas = np.zeros((H_raw, W_raw, 3), dtype=np.uint8)
        canvas[:, :] = (8, 12, 18)

        # Colores por tipo funcional de unidad en el Raster Plot
        c_map = np.array([
            [200, 210, 220],  # 0: Interna
            [88, 166, 255],   # 1: Sensorial (azul)
            [255, 166, 87],   # 2: Motora Derecha (naranja)
            [57, 211, 83],    # 3: Motora Izquierda (verde)
        ], dtype=np.uint8)

        row_colors = c_map[s.rasterTypes][:, None, :]
        active_mask = s.rasterHist != 0
        canvas = np.where(active_mask[:, :, None], row_colors, canvas)

        # Escalar verticalmente x2 (200 px de alto)
        img = np.repeat(canvas, 2, axis=0)
        return img

    def _draw_firing_rates(self):
        """Dibuja las curvas temporales de Tasa de Disparo Poblacional (LFP sintético)."""
        cv = self.cvRate
        cv.delete('all')
        W, H = NW_W, 170

        # Líneas guía de referencia (25%, 50%, 75%)
        for frac, lbl in ((0.25, '0.25'), (0.50, '0.50'), (0.75, '0.75')):
            y = H - int(frac * (H - 20)) - 10
            cv.create_line(0, y, W, y, fill='#1b222c', width=1)
            cv.create_text(4, y - 6, text=lbl, fill=MUTED, anchor='w', font=('Menlo', 8))

        xs = np.linspace(0, W - 1, HIST_LEN)

        def plot_curve(arr, color, width=1.5):
            ys = H - 10 - np.clip(arr, 0.0, 1.0) * (H - 24)
            pts = []
            for x, y in zip(xs, ys):
                pts.extend((float(x), float(y)))
            if len(pts) >= 4:
                cv.create_line(*pts, fill=color, width=width)

        plot_curve(self.sim.rateGlobalHist, '#8b949e', width=1)
        plot_curve(self.sim.rateDerHist, '#ffa657', width=2)
        plot_curve(self.sim.rateIzqHist, '#39d353', width=2)

        cv.create_text(W - 8, 12, text='— Mot Der (f1)   — Mot Izq (f2)   — Global',
                       fill=TXT, anchor='e', font=('Helvetica', 8))

    def draw(self):
        s = self.sim
        cv = self.cvA
        if self._arena_bg is None or s._obs_dirty:
            self.imgA = self._photo(self._arena_image())
            cv.itemconfig(self.itemA, image=self.imgA)
            self._arena_bg = True
            s._obs_dirty = False
        cv.delete('ov')

        if s.arenaShape == 'circular':
            r_px = RADIO_ARENA * AS
            cx_px, cy_px = CENTRO_ARENA[0] * AS, CENTRO_ARENA[1] * AS
            cv.create_oval(cx_px - r_px, cy_px - r_px, cx_px + r_px, cy_px + r_px,
                           outline='#30363d', width=2, tags='ov')

        if len(s.trail) > 1:
            pts = []
            for (x, y) in s.trail:
                pts.extend((x * AS, y * AS))
            cv.create_line(*pts, fill='#2f6fa8', width=1, tags='ov')

        for a in range(NRAY):
            ang = s.carDir + SENSOR_ANGLES[a]
            d = s.lastDist[a]
            ex = s.carX + math.cos(ang) * d
            ey = s.carY + math.sin(ang) * d
            inten = max(0.0, 1 - d / MAXR)
            g = int(60 + 150 * inten)
            cv.create_line(s.carX * AS, s.carY * AS, ex * AS, ey * AS,
                           fill='#%02x%02x%02x' % (g // 3, g, 255), width=1, tags='ov')

        cd, sd = math.cos(s.carDir), math.sin(s.carDir)
        pts = []
        for (lx, ly) in ((CAR_R * 1.3, 0), (-CAR_R, CAR_R * 0.8), (-CAR_R, -CAR_R * 0.8)):
            pts.extend(((s.carX + lx * cd - ly * sd) * AS, (s.carY + lx * sd + ly * cd) * AS))
        cv.create_polygon(*pts, fill=ACCENT, outline='', tags='ov')

        if s.flash > 0:
            k = s.flash / 6.0
            rr = CAR_R * AS * (2 + 2 * (1 - k))
            cv.create_oval(s.carX * AS - rr, s.carY * AS - rr, s.carX * AS + rr, s.carY * AS + rr,
                           outline=s.flashColor, width=2, tags='ov')
            s.flash -= 1

        self.imgB = self._photo(self._brain_image())
        self.cvB.itemconfig(self.itemB, image=self.imgB)

        self.imgRaster = self._photo(self._raster_image())
        self.cvRaster.itemconfig(self.itemRaster, image=self.imgRaster)
        self._draw_firing_rates()

    def update_hud(self):
        self.hud.config(text=self.sim.hud_text())
        self.tramosLbl.config(text=self.sim.texto_tramos())
        if not self._busy():
            self.ffstatus.config(text='paso %d' % self.sim.step)


# =====================================================================
# 7. EJECUCIÓN HEADLESS (PARA RANKING DEL DÍA 3) Y PUNTO DE ENTRADA
# =====================================================================
def run_headless(sim, steps, report_every=5000):
    print(
        f"semilla={sim.seedValue} | modo={sim.brainMode} | arena={sim.arenaShape} | "
        f"cableado={sim.wiring} | feedback={'ON' if sim.sensFeedback else 'OFF'}"
    )
    for _ in range(steps):
        sim.advance()
        if report_every and sim.step % report_every == 0:
            act_pct = 100.0 * float(np.count_nonzero(sim.grid)) / float(M)
            print(
                f"  paso {sim.step:6d} | choques obj={sim.crashObj:4d} borde={sim.crashBorder:4d} | "
                f"dist={sim.distTraveled:9.0f} | activas={act_pct:5.1f}%"
            )
    print("\n" + sim.hud_text())
    print("\n" + sim.texto_tramos())


def main():
    ap = argparse.ArgumentParser(description='Código Base Bootcamp IA Bioinspirada (Conway + Carro)')
    ap.add_argument('--headless', type=int, metavar='PASOS',
                    help='ejecuta N pasos por consola sin GUI e imprime el ranking por tramos')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--mode', default='conway_prob',
                    choices=['conway_clasico', 'conway_prob', 'bio_plastico', 'metabolico_40k'],
                    help='regla local del autómata')
    ap.add_argument('--arena', default='cuadrada', choices=['cuadrada', 'circular'],
                    help='geometría de la arena de navegación')
    ap.add_argument('--wiring', default='ipsi', choices=['ipsi', 'contra'])
    ap.add_argument('--no-feedback', action='store_true')
    ap.add_argument('--controller', default='automata',
                    choices=['automata', 'braitAvoid', 'braitSeek', 'braitOsc', 'levy'])
    ap.add_argument('--field', default='mixto',
                    choices=['mixto', 'todas', 'cuaddiag', 'diag', 'cuad', 'cruz', 'bananas'])
    args = ap.parse_args()

    sim = BootcampSim()
    sim.seedValue = args.seed
    sim.brainMode = args.mode
    sim.arenaShape = args.arena
    sim.wiring = args.wiring
    sim.controller = args.controller
    sim.fieldType = args.field
    if args.no_feedback:
        sim.sensFeedback = False
    sim.reset()

    if args.headless:
        run_headless(sim, args.headless)
        return

    try:
        import tkinter as tk
    except ImportError:
        print('tkinter no disponible; ejecuta con --headless N', file=sys.stderr)
        sys.exit(1)
    root = tk.Tk()
    App(root, sim)
    root.mainloop()


if __name__ == '__main__':
    main()

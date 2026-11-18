#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plantilla de Trabajo para Equipos — Bootcamp IA Bioinspirada
Importa el motor base y sobreescribe únicamente las reglas locales.
"""

import numpy as np
import bootcamp_conway_carro as base


class MiCerebroBio(base.BootcampSim):

    def paso_automata_local(self, t, extStim):
        """
        Edita aquí la regla local de tu dupla (Ingeniería + Biología).
        Variables disponibles:
          - self.grid       : vector 1D (1600,) con estado 0 o 1
          - self.energia    : vector 1D (1600,) con reserva de ATP (0..300)
          - self.umbralLocal: matriz 2D (40, 40) con el umbral de cada celda
          - extStim         : vector 1D (1600,) con los pulsos sensoriales (0 o 1)
        """
        G = self.grid.reshape(base.N, base.N)
        ext2D = (extStim != 0).reshape(base.N, base.N)

        # 1. Contar vecinos activos (Vecindario de Moore 8 vecinos)
        vecinos = self._contar_vecinos_moore(G)

        # 2. Regla probabilística inspirada en biología
        R = self.rng.random_matrix((base.N, base.N))
        nace = (G == 0) & (vecinos == 3) & (R < self.pBirth)
        sobrevive = (G == 1) & ((vecinos == 2) | (vecinos == 3)) & (R < self.pSurvive)

        nueva_grilla = nace | sobrevive | ext2D
        self.grid = nueva_grilla.astype(np.uint8).ravel()


if __name__ == '__main__':
    sim = MiCerebroBio()
    sim.arenaShape = 'circular'   # o 'cuadrada'
    sim.reset()

    import tkinter as tk
    root = tk.Tk()
    base.App(root, sim)
    root.mainloop()
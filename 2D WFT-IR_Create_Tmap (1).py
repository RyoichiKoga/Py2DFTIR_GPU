#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:39:02 2026

@author: user
"""

import numpy as np

frames = 2000
N1, N2 = frames, 8192
pix_size = 17e3      # 像素尺寸 (nm)
f = 61e6             # 焦距 (nm)
M_L = 1.99905        # 可动镜行程 (mm)
M = M_L * 1e6 / N1   # nm per frame

phi_y = 57.642 * np.pi / 180.0
cx, cy = 319, 239    # 中心像素坐标

window = np.hanning(N1)           # 或 np.hanning(N1)
wnum = np.arange(495, 2500, 4, dtype=np.float32)  # 波数轴 (cm⁻¹)
nw = len(wnum)

nx, ny = 640,480

def compute_T_map(ny, nx, cx, cy, pix_size, f, phi_y, M):
    """
    各ピクセルのサンプリング間隔T（秒）を作る (ny, nx)
    """
    py = np.arange(ny)[:, None]
    px = np.arange(nx)[None, :]

    a = -(px - cx) * pix_size
    b = (py - cy) * pix_size

    thx = np.arctan(b / f)
    thy = np.arctan(a / f)
    thd = thy + np.pi/2

    L = 2 * M * np.cos(phi_y - thd) / np.cos(thx)  # nm
    T = L * 1e-7                                   # seconds
    return T

# 前計算
T_map = compute_T_map(ny, nx, cx, cy, pix_size, f, phi_y, M)

meta = dict(
    ny=ny, nx=nx, cx=cx, cy=cy,
    pix_size=pix_size, f=f, phi_y=phi_y, M=M,
    note="T_map for this optical geometry"
)

np.savez("T_map_lookup.npz", T_map=T_map, meta=meta)

import numpy as np
import matplotlib.pyplot as plt

# 読み込み
dat = np.load("T_map_lookup.npz", allow_pickle=True)
T_map = dat["T_map"]

print("shape:", T_map.shape)
print("min/max:", T_map.min(), T_map.max())

plt.figure(figsize=(6,5))
cs = plt.contourf(T_map, levels=50)
plt.colorbar(cs, label="T_map value")
plt.title("T_map contour")
plt.xlabel("x pixel")
plt.ylabel("y pixel")
plt.tight_layout()
plt.show()

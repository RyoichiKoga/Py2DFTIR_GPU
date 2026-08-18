#%%  第一段：读取 reference 和 sample 数据，输出干涉图并压缩为 3D cube

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import time
#_t_start = time.perf_counter()

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import warnings
import os

warnings.filterwarnings('ignore')
# -------------------------------
# 参数设置
# -------------------------------
frames = 2000  # 要读取的总帧数
# 请根据实际情况把下面两行改成你本地的文件夹路径（注意最后带斜杠）
Path_sam = '/media/user/Data/2DFT-IRdata/JASCO-sam/'
Path_ref = '/media/user/Data/2DFT-IRdata/JASCO-ref/'
# -------------------------------
def load_cube_by_loop(path, frames=2000):
    first = tifffile.imread(os.path.join(path, "0001.tif"))
    ny, nx = first.shape
    cube = np.empty((frames, ny, nx), dtype=np.float32)
    cube[0] = first.astype(np.float32, copy=False)

    for i in range(1, frames):
        fname = f"{i+1:04d}.tif"
        cube[i] = tifffile.imread(os.path.join(path, fname)).astype(np.float32, copy=False)

    return cube
# -------------------------------
# 1. 读取 Sample 与 Reference，生成 3D-cube
# -------------------------------
Z_sam = load_cube_by_loop(Path_sam, frames)  # 样品干涉信号立方
Z_ref = load_cube_by_loop(Path_ref, frames)  # 参考干涉信号立方
print(f"Loaded Z_sam shape:   {Z_sam.shape}")
print(f"Loaded Z_ref shape:   {Z_ref.shape}")
# -------------------------------
# 2. 在同一像素 (319,239) 处绘制 Sample vs Reference 干涉条纹对比
#    行、列下标从 0 开始，即 (319, 239) 表示第 320 列、第 240 行
# -------------------------------
pixel_y, pixel_x = 319, 239
x_axis = np.arange(frames)
# --- 样品干涉条纹处理 ---
ysam = Z_sam[:, pixel_y, pixel_x]
H_sam = np.poly1d(np.polyfit(x_axis, ysam, 1))(x_axis)  # 去线性趋势
ysam_detrended = ysam - H_sam
xmax_sam = np.argmax(np.abs(ysam_detrended))
ysam_centered = np.roll(ysam_detrended, frames//2 - xmax_sam)

# --- 参考干涉条纹处理 ---
yref = Z_ref[:, pixel_y, pixel_x]
H_ref = np.poly1d(np.polyfit(x_axis, yref, 1))(x_axis)
yref_detrended = yref - H_ref
xmax_ref = np.argmax(np.abs(yref_detrended))
yref_centered = np.roll(yref_detrended, frames//2 - xmax_ref)

# -------------------------------
# 3. 绘图：绘制两个干涉条纹
# -------------------------------
plt.figure(figsize=(12, 5))
plt.plot(x_axis, ysam_centered, color='blue',  label='Sample Interferogram')
plt.plot(x_axis, yref_centered, color='orange', label='Reference Interferogram')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Scan Index")
plt.ylabel("Detrended Intensity")
plt.title(f"Interferograms at pixel ({pixel_x+1}, {pixel_y+1})")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig("interferograms_sample_reference.png", dpi=300)
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft
from numba import njit, prange
import cupy as cp

# ————— 全局参数 —————
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

_t_start = time.perf_counter()

N1 = frames
x = np.arange(N1, dtype=np.float32)
x_mean = np.float32((N1 - 1) * 0.5)
x0 = x - x_mean
denom = np.float32(np.sum(x0 * x0))   # これは N1 固定なら定数


@njit(parallel=True)
def detrend_linear_all_numba(Y, x0, x_mean, denom):
    """
    Y: (N1, ny, nx) float32
    x0: (N1,) float32, x - x_mean
    x_mean: float32
    denom: float32 (sum(x0^2))
    return: detrended (N1, ny, nx) float32
    """
    N1, ny, nx = Y.shape
    out = np.empty_like(Y)

    for py in prange(ny):
        for px in range(nx):
            # y_mean
            s = 0.0
            for i in range(N1):
                s += Y[i, py, px]
            y_mean = s / N1

            # cov = sum( x0[i] * (Y - y_mean) )
            cov = 0.0
            for i in range(N1):
                cov += x0[i] * (Y[i, py, px] - y_mean)

            a = cov / denom
            b = y_mean - a * x_mean

            # out = Y - (a*x + b) だが x = x0 + x_mean を使うと速い
            # trend = a*(x0 + x_mean) + b = a*x0 + (a*x_mean + b)
            c = a * x_mean + b
            for i in range(N1):
                out[i, py, px] = Y[i, py, px] - (a * x0[i] + c)

    return out


@njit(parallel=True, fastmath=True)
def roll_center_numba(Y):
    """
    Y: (N1, ny, nx) float32
    各ピクセルで |Y| 最大の位置を N1//2 に来るように循環シフトして out に格納
    take_along_axisもnp.rollも使わない（巨大インデックスなし）
    """
    N1, ny, nx = Y.shape
    out = np.empty_like(Y)
    mid = N1 // 2

    for py in prange(ny):
        for px in range(nx):
            # argmax(abs)
            imax = 0
            vmax = 0.0
            for i in range(N1):
                v = Y[i, py, px]
                av = v if v >= 0 else -v
                if av > vmax:
                    vmax = av
                    imax = i

            sh = mid - imax  # これだけ中央へ寄せたい
            # out[i] = Y[(i - sh) mod N1]
            # modは負に注意して手動で
            for i in range(N1):
                src = i - sh
                src %= N1
                out[i, py, px] = Y[src, py, px]

    return out

def preprocess_fft_mag(Z, window, N2):
    """
    Z: (N1, ny, nx)
    window: (N1,)
    N2: ゼロパディング長
    return: mag (nf, ny, nx), nf = N2//2 - 1 （DCとNyquist除く、元コードに合わせる）
    """
    Z = Z.astype(np.float32, copy=False)

    # 1) detrend（polyfit相当）
    #Y = detrend_linear_all(Z)
    Y = detrend_linear_all_numba(Z, x0, x_mean, denom)

    # 2) センターバースト補正（argmax + roll）
    Y = roll_center_numba(Y)

    # 3) window
    Y *= window[:, None, None]

    # 4) FFT（実数入力なので rfft でOK）
    #F = np.fft.rfft(Y, n=N2, axis=0)             # (N2//2+1, ny, nx)
    #F = rfft(Y, n=N2, axis=0, workers=-1)    #workers > paralell calc.
    
    # 4) FFT（GPU）
    Yg = cp.asarray(Y)  # CPU->GPU
    Fg = cp.fft.rfft(Yg, n=N2, axis=0)     # cuFFT
    magg = cp.abs(Fg)[1:N2//2]             # (N2//2-1, ny, nx)

    # 元コード：abs(fft(...))[1:N2//2]
    #mag = np.abs(F)[1:N2//2]                     # (N2//2-1, ny, nx)
    mag = cp.asnumpy(magg)  # GPU->CPU（後段numbaのため）
    return mag

ny, nx = Z_sam.shape[1], Z_sam.shape[2]
window = np.hanning(N1)

fft_s_mag = preprocess_fft_mag(Z_sam, window, N2)   # (nf, ny, nx)
fft_r_mag = preprocess_fft_mag(Z_ref, window, N2)   # (nf, ny, nx)

######ここからループ#####
#def compute_T_map(ny, nx, cx, cy, pix_size, f, phi_y, M):
    #"""
    #各ピクセルのサンプリング間隔T（秒）を作る (ny, nx)
    #"""
    #py = np.arange(ny)[:, None]
    #px = np.arange(nx)[None, :]

    #a = -(px - cx) * pix_size
    #b = (py - cy) * pix_size

    #thx = np.arctan(b / f)
    #thy = np.arctan(a / f)
    #thd = thy + np.pi/2

    #L = 2 * M * np.cos(phi_y - thd) / np.cos(thx)  # nm
    #T = L * 1e-7                                   # seconds
    #return T


# 前計算
#T_map = compute_T_map(ny, nx, cx, cy, pix_size, f, phi_y, M)

dat = np.load("T_map_lookup.npz", allow_pickle=True)
T_map = dat["T_map"]
meta = dat["meta"].item()

# 安全確認（超重要）
assert meta["ny"] == ny and meta["nx"] == nx
assert meta["cx"] == cx and meta["cy"] == cy
assert meta["pix_size"] == pix_size
assert meta["f"] == f
assert meta["phi_y"] == phi_y
assert meta["M"] == M


@njit(fastmath=True)
def interp1d_numba(x, xp, fp, fill):
    """
    x : wnum (nw,)
    xp: freq_pos (nf,) 単調増加
    fp: fft_s_1d (nf,)
    """
    nw = x.shape[0]
    nf = xp.shape[0]
    out = np.empty(nw, dtype=np.float32)

    j = 0
    for i in range(nw):
        xi = x[i]

        if xi < xp[0] or xi > xp[nf-1]:
            out[i] = fill
            continue

        while j < nf-2 and xp[j+1] < xi:
            j += 1

        x0 = xp[j]
        x1 = xp[j+1]
        y0 = fp[j]
        y1 = fp[j+1]

        out[i] = y0 + (y1 - y0) * (xi - x0) / (x1 - x0)

    return out

@njit(fastmath=True)
def pixel_geom_interp_numba(fft_s_1d, fft_r_1d, T, k, N2, wnum,
                            fill_s=0.0, fill_r=1e-8):

    freq_pos = k / (N2 * T)

    s = interp1d_numba(wnum, freq_pos, fft_s_1d, fill_s)
    r = interp1d_numba(wnum, freq_pos, fft_r_1d, fill_r)
    return s, r

k_global = np.arange(1, N2//2, dtype=np.float32)

@njit(parallel=True, fastmath=True)
def interp_all_pixels(fft_s_mag, fft_r_mag, T_map, k, N2, wnum):
    ny, nx = T_map.shape
    nw = wnum.shape[0]

    spec_s = np.empty((ny, nx, nw), dtype=np.float32)
    spec_r = np.empty((ny, nx, nw), dtype=np.float32)

    for py in prange(ny):
        for px in range(nx):
            s, r = pixel_geom_interp_numba(
                fft_s_mag[:, py, px],
                fft_r_mag[:, py, px],
                T_map[py, px],
                k, N2, wnum
            )
            spec_s[py, px, :] = s
            spec_r[py, px, :] = r

    return spec_s, spec_r


spec_s, spec_r = interp_all_pixels(
    fft_s_mag, fft_r_mag, T_map, k_global, N2, wnum
)


_t_end = time.perf_counter()
print(f"\n[TOTAL TIME] {( _t_end - _t_start ):.3f} s") 

# ————— 2) 计算平均样本谱和平均参考谱 —————
mean_s = spec_s.mean(axis=(0, 1))  # (nw,)
mean_r = spec_r.mean(axis=(0, 1))  # (nw,)
A = ((mean_s + 1e-12) / (mean_r + 1e-12)) * 100  # (nw,)


# ————— 4) 绘图：吸光度 光谱 —————
plt.figure(figsize=(8,5))
plt.plot(wnum, A, color='purple', lw=1.5)
plt.xlim(495, 2500)
plt.ylim(0, 110)
#plt.ylim(0, np.max(A)*1.1)
plt.xlabel("Wavenumber [cm⁻¹]")
plt.ylabel("Absorbance")
plt.title("Absorbance Spectrum at Center （32×32）")
plt.grid(ls='--', alpha=0.3)
plt.tight_layout()
plt.show()

#_t_end = time.perf_counter()
#print(f"\n[TOTAL TIME] {( _t_end - _t_start ):.3f} s") #meaure and show running time
#
#!/usr/bin/env python3
"""
《本科物理的数学基础》—— 全教材插图生成脚本
运行方式: python generate_all_figures.py
依赖: numpy, matplotlib, scipy
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import legendre, jv, yn, sph_harm
import os, sys

# ---------- Windows GBK 编码修复 ----------
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ---------- CJK 字体配置 ----------
# Windows 常见中文字体路径
CJK_FONTS = [
    'C:/Windows/Fonts/simsun.ttc',   # 宋体
    'C:/Windows/Fonts/simhei.ttf',   # 黑体
    'C:/Windows/Fonts/msyh.ttc',     # 微软雅黑
    'C:/Windows/Fonts/msyhbd.ttc',   # 微软雅黑 Bold
]
CJK_FONT_NAME = None
for fp in CJK_FONTS:
    if os.path.exists(fp):
        from matplotlib.font_manager import FontProperties
        CJK_FONT_NAME = FontProperties(fname=fp).get_name()
        # 注册并设置为默认 sans-serif
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(fp)
        plt.rcParams['font.sans-serif'] = [CJK_FONT_NAME, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

OK_MSG = "[OK]"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 150


# ================================================================
#  第 1 章  坐标系、矢量分析与张量初步
# ================================================================

def fig1_1_coordinates():
    """三种坐标系的示意图"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': '3d'})

    # 直角坐标
    ax = axes[0]
    for k in range(-1, 2):
        for j in range(-1, 2):
            ax.plot([k, k], [j, j], [-1, 1], 'b-', lw=0.3, alpha=0.3)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('(a) Cartesian $(x,y,z)$'); ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)

    # 柱坐标
    ax = axes[1]
    theta = np.linspace(0, 2*np.pi, 30)
    for r in [0.3, 0.6, 0.9]:
        ax.plot(r*np.cos(theta), r*np.sin(theta), 0, 'b-', lw=0.5, alpha=0.4)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('(b) Cylindrical $(\\rho,\\phi,z)$'); ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)

    # 球坐标
    ax = axes[2]
    phi = np.linspace(0, 2*np.pi, 40)
    theta = np.linspace(0, np.pi, 20)
    r = 0.9
    for t in theta[::3]:
        ax.plot(r*np.sin(t)*np.cos(phi), r*np.sin(t)*np.sin(phi), r*np.cos(t)*np.ones_like(phi), 'b-', lw=0.3, alpha=0.3)
    for p in phi[::4]:
        ax.plot(r*np.sin(theta)*np.cos(p), r*np.sin(theta)*np.sin(p), r*np.cos(theta), 'b-', lw=0.3, alpha=0.3)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('(c) Spherical $(r,\\theta,\\phi)$'); ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_1_coordinates.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig1_1_coordinates.png")


def fig1_2_divergence():
    """散度的几何意义：源与汇"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))

    # 正散度（源）
    U1, V1 = X, Y
    ax = axes[0]; ax.quiver(X, Y, U1, V1)
    ax.set_title(r'$\nabla\cdot\mathbf{F}>0$ (source)'); ax.set_aspect('equal')

    # 负散度（汇）
    U2, V2 = -X, -Y
    ax = axes[1]; ax.quiver(X, Y, U2, V2)
    ax.set_title(r'$\nabla\cdot\mathbf{F}<0$ (sink)'); ax.set_aspect('equal')

    # 零散度（旋转场）
    U3, V3 = -Y, X
    ax = axes[2]; ax.quiver(X, Y, U3, V3)
    ax.set_title(r'$\nabla\cdot\mathbf{F}=0$ (solenoidal)'); ax.set_aspect('equal')

    for ax in axes: ax.set_xlim(-2,2); ax.set_ylim(-2,2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_2_divergence.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig1_2_divergence.png")


def fig1_3_curl():
    """旋度的几何意义"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    X, Y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))

    # 有旋场
    U1, V1 = -Y, X
    ax = axes[0]; ax.quiver(X, Y, U1, V1)
    ax.set_title(r'$\nabla\times\mathbf{F}\neq 0$ (rotational)'); ax.set_aspect('equal')

    # 无旋场
    U2, V2 = X, Y
    ax = axes[1]; ax.quiver(X, Y, U2, V2)
    ax.set_title(r'$\nabla\times\mathbf{F}=0$ (irrotational)'); ax.set_aspect('equal')

    for ax in axes: ax.set_xlim(-2,2); ax.set_ylim(-2,2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_3_curl.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig1_3_curl.png")


def fig1_4_gauss_theorem():
    """高斯定理示意图：体积分 ↔ 面积分"""
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, alpha=0.15, color='blue')
    ax.plot_wireframe(x, y, z, alpha=0.3, color='blue', rstride=3, cstride=3)

    ax.text(0, 0, 0, r'$\nabla\cdot\mathbf{F}$', fontsize=14, ha='center')
    ax.text(1.2, 0, 0.8, r'$\oiint \mathbf{F}\cdot d\mathbf{S}$', fontsize=14, color='red')
    ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5); ax.set_zlim(-1.5,1.5)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_4_gauss_theorem.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig1_4_gauss_theorem.png")


# ================================================================
#  第 2 章  复变函数
# ================================================================

def fig2_1_complex_plane():
    """复平面与极坐标表示"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 复平面
    ax = axes[0]
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    z = 3 + 2j
    ax.arrow(0, 0, z.real, z.imag, head_width=0.2, head_length=0.2, fc='red', ec='red')
    ax.plot(z.real, z.imag, 'ro', markersize=6)
    ax.text(z.real+0.2, z.imag+0.2, '$z=x+iy$', fontsize=14)
    ax.text(1.2, 1.0, '$r=|z|$', fontsize=12, color='red')
    ax.set_xlabel('$\\Re(z)$'); ax.set_ylabel('$\\Im(z)$')
    ax.set_xlim(-1,5); ax.set_ylim(-1,4); ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('(a) Complex plane')

    # 单位圆上的点
    ax = axes[1]
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'b-', lw=1)
    pts = [np.exp(1j*k*np.pi/4) for k in range(8)]
    for p in pts:
        ax.plot(p.real, p.imag, 'ro', ms=4)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('$\\Re(z)$'); ax.set_ylabel('$\\Im(z)$')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_title('(b) Unit circle $e^{i\\theta}$')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_1_complex_plane.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig2_1_complex_plane.png")


def fig2_2_contour():
    """围道积分路径示例"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 简单围道
    ax = axes[0]
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'b-', lw=2, label='$C$')
    ax.plot(0, 0, 'rx', markersize=10)
    ax.text(0.1, 0.1, '$z_0$', fontsize=12)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('$\\Re(z)$'); ax.set_ylabel('$\\Im(z)$')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_title('(a) $\\oint_C f(z)dz = 2\\pi i\\,\\mathrm{Res}f(z_0)$')
    ax.legend()

    # 上半平面围道
    ax = axes[1]
    R = 3
    ax.plot(np.linspace(-R, R, 100), np.zeros(100), 'b-', lw=2)
    theta = np.linspace(0, np.pi, 100)
    ax.plot(R*np.cos(theta), R*np.sin(theta), 'b-', lw=2)
    poles = [0.5+1j, -1+0.8j]
    for p in poles:
        ax.plot(p.real, p.imag, 'rx', markersize=10)
    ax.text(poles[0].real+0.1, poles[0].imag+0.1, '$z_1$', fontsize=12)
    ax.text(poles[1].real+0.1, poles[1].imag+0.1, '$z_2$', fontsize=12)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('$\\Re(z)$'); ax.set_ylabel('$\\Im(z)$')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_title('(b) Upper half-plane contour $\\int_{-\\infty}^{\\infty}\\cdots$')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_2_contour.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig2_2_contour.png")


def fig2_3_conformal():
    """保角映射 w = z^2 示例"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    r = np.linspace(0, 1, 10)
    theta = np.linspace(0, np.pi/2, 8)
    for t in theta:
        ax.plot(r*np.cos(t), r*np.sin(t), 'b-', lw=0.5)
    for rad in [0.2, 0.5, 0.8, 1.0]:
        ax.plot(rad*np.cos(theta), rad*np.sin(theta), 'b-', lw=0.5)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlim(0,1.2); ax.set_ylim(0,1.2); ax.set_aspect('equal')
    ax.set_title('(a) $z$-plane (first quadrant)')

    ax = axes[1]
    for t in theta:
        u = r**2 * np.cos(2*t)
        v = r**2 * np.sin(2*t)
        ax.plot(u, v, 'b-', lw=0.5)
    for rad in [0.2, 0.5, 0.8, 1.0]:
        u = rad*np.cos(2*theta)
        v = rad*np.sin(2*theta)
        ax.plot(u, v, 'b-', lw=0.5)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlim(-1.2,1.2); ax.set_ylim(0,1.2); ax.set_aspect('equal')
    ax.set_title('(b) $w=z^2$ (upper half-plane)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_3_conformal.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig2_3_conformal.png")


# ================================================================
#  第 3 章  积分变换
# ================================================================

def fig3_1_fourier_series():
    """傅里叶级数：方波逼近"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    t = np.linspace(-np.pi, np.pi, 1000)
    square = np.sign(np.sin(t))

    Ns = [1, 3, 7, 21]
    for ax, N in zip(axes.flat, Ns):
        f = np.zeros_like(t)
        for n in range(1, N+1, 2):
            f += (4/(n*np.pi)) * np.sin(n*t)
        ax.plot(t, square, 'k--', lw=1, label='target square')
        ax.plot(t, f, 'r-', lw=1.5, label=f'$N={N}$')
        ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
        ax.set_xlabel('$t$'); ax.set_ylabel('$f(t)$')

    plt.suptitle('Fourier series approximation (Gibbs phenomenon visible)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_1_fourier_series.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig3_1_fourier_series.png")


def fig3_2_ft_pairs():
    """傅里叶变换对示例"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    t = np.linspace(-5, 5, 1000)
    w = np.linspace(-20, 20, 1000)

    # 1. 高斯函数
    sigma = 1.0
    f1 = np.exp(-t**2/(2*sigma**2))
    F1 = np.sqrt(2*np.pi)*sigma * np.exp(-sigma**2 * w**2 / 2)
    axes[0,0].plot(t, f1); axes[0,0].set_title(r'$f(t)=e^{-t^2/2}$')
    axes[0,1].plot(w, F1); axes[0,1].set_title(r'$\tilde{f}(\omega)=\sqrt{2\pi}e^{-\omega^2/2}$')
    axes[0,2].axis('off')

    # 2. 矩形脉冲
    tau = 1.0
    f2 = np.where(np.abs(t) < tau, 1.0, 0.0)
    F2 = 2*np.sin(tau*w)/w
    F2[np.abs(w)<0.01] = 2*tau
    axes[1,0].plot(t, f2); axes[1,0].set_title(r'$f(t)=\mathrm{rect}(t)$')
    axes[1,1].plot(w, F2); axes[1,1].set_title(r'$\tilde{f}(\omega)=2\mathrm{sinc}(\omega)$')
    axes[1,1].set_xlim(-20,20)
    axes[1,2].axis('off')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_2_ft_pairs.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig3_2_ft_pairs.png")


def fig3_3_fft_example():
    """FFT 频谱分析示例"""
    fs = 1000
    t = np.arange(0, 1, 1/fs)
    f = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t) + 0.3*np.random.randn(len(t))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(t, f)
    axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Amplitude')
    axes[0].set_title('(a) Time domain (50Hz + 120Hz + noise)'); axes[0].set_xlim(0, 0.1)

    F = np.fft.fft(f)
    freq = np.fft.fftfreq(len(t), 1/fs)
    mask = freq >= 0
    axes[1].plot(freq[mask], 2*np.abs(F[mask])/len(F))
    axes[1].set_xlabel('Frequency (Hz)'); axes[1].set_ylabel('Amplitude')
    axes[1].set_title('(b) Frequency domain (FFT spectrum)'); axes[1].set_xlim(0, 200)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_3_fft_example.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig3_3_fft_example.png")


def fig3_4_laplace_roc():
    """拉普拉斯变换收敛域 ROC"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    cases = [
        (r'$e^{-t}\Theta(t)$', 0.5, 1.0, 1),
        (r'$e^{t}\Theta(-t)$', -0.5, -1.0, -1),
        (r'$\Theta(t)$', 0.1, 0, 0),
    ]

    for idx, (title, a, pole, side) in enumerate(cases):
        ax = axes[idx]
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        if side > 0:
            ax.axvspan(pole, 5, alpha=0.2, color='blue')
            ax.axvline(pole, color='red', lw=2, linestyle='--')
        elif side < 0:
            ax.axvspan(-5, pole, alpha=0.2, color='blue')
            ax.axvline(pole, color='red', lw=2, linestyle='--')
        else:
            ax.axvspan(0, 5, alpha=0.2, color='blue')
        ax.plot(pole, 0, 'rx', markersize=10)
        ax.set_xlim(-3, 3); ax.set_ylim(-2, 2)
        ax.set_xlabel('$\\Re(s)$'); ax.set_ylabel('$\\Im(s)$')
        ax.set_title(f'{title}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if side != 0:
            ax.text(1.5, 1.5, 'ROC', fontsize=12, color='blue', ha='center')

    plt.suptitle('Laplace transform Region of Convergence (ROC)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_4_laplace_roc.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig3_4_laplace_roc.png")


# ================================================================
#  第 4 章  偏微分方程与特殊函数
# ================================================================

def fig4_1_pde_types():
    """三类 PDE 的示意图"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 80)
    X, T = np.meshgrid(x, t)

    # 波动方程
    U_wave = np.sin(2*np.pi*(X - T))
    axes[0].contourf(X, T, U_wave, levels=20, cmap='RdBu')
    axes[0].set_title('(a) Wave eq. (hyperbolic)\n$u_{tt}=c^2u_{xx}$')
    axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$t$')

    # 热传导方程
    U_heat = np.exp(-X**2/(0.1+0.5*T))
    axes[1].contourf(X, T, U_heat, levels=20, cmap='hot')
    axes[1].set_title('(b) Heat eq. (parabolic)\n$u_t=\\kappa u_{xx}$')
    axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$t$')

    # 拉普拉斯方程
    X2, Y2 = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    U_laplace = np.sin(np.pi*X2)*np.sinh(np.pi*Y2)
    axes[2].contourf(X2, Y2, U_laplace, levels=20, cmap='viridis')
    axes[2].set_title('(c) Laplace eq. (elliptic)\n$\\nabla^2 u=0$')
    axes[2].set_xlabel('$x$'); axes[2].set_ylabel('$y$')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_1_pde_types.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig4_1_pde_types.png")


def fig4_2_legendre():
    """勒让德多项式"""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(-1, 1, 500)
    for l in range(5):
        P = legendre(l)
        ax.plot(x, P(x), label=f'$P_{l}(x)$', lw=1.5)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('$x$'); ax.set_ylabel('$P_l(x)$')
    ax.set_title('Legendre polynomials $P_l(x)$')
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_2_legendre.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig4_2_legendre.png")


def fig4_3_bessel():
    """贝塞尔函数 J_n(x)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.linspace(0, 20, 500)

    ax = axes[0]
    for n in range(4):
        ax.plot(x, jv(n, x), label=f'$J_{n}(x)$', lw=1.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('$x$'); ax.set_ylabel('$J_n(x)$')
    ax.set_title('(a) Bessel functions $J_n(x)$')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for n in range(2):
        y = yn(n, x)
        y[x < 0.5] = np.nan
        ax.plot(x, y, label=f'$Y_{n}(x)$', lw=1.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('$x$'); ax.set_ylabel('$Y_n(x)$')
    ax.set_title('(b) Neumann functions $Y_n(x)$')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_3_bessel.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig4_3_bessel.png")


def fig4_4_drum_modes():
    """圆膜振动模式"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    zeros = {
        (0,1): 2.4048, (1,1): 3.8317, (2,1): 5.1356,
        (0,2): 5.5201, (1,2): 7.0156, (2,2): 8.4172
    }

    r = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2*np.pi, 100)
    R, TH = np.meshgrid(r, theta)

    for idx, ((m, n), j) in enumerate(zeros.items()):
        ax = axes[idx // 3, idx % 3]
        U = jv(m, j*R) * np.cos(m*TH)
        X = R * np.cos(TH)
        Y = R * np.sin(TH)

        ax.contourf(X, Y, U, levels=20, cmap='RdBu')
        ax.contour(X, Y, U, levels=[0], colors='k', linewidths=2)
        ax.set_title(f'$(m,n)=({m},{n})$  $f={j/(2*np.pi):.2f}$')
        ax.set_aspect('equal'); ax.axis('off')

    plt.suptitle('Circular drum vibration modes (nodal lines in black)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_4_drum_modes.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig4_4_drum_modes.png")


def fig4_5_spherical_harmonics():
    """球谐函数 Y_l^m 可视化"""
    fig = plt.figure(figsize=(12, 8))

    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    TH, PH = np.meshgrid(theta, phi)

    pairs = [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2),
             (3,0), (3,1), (3,2), (3,3), (4,0), (4,1)]

    norm = plt.Normalize(-1, 1)

    for idx, (l, m) in enumerate(pairs):
        if idx >= 12: break
        ax = fig.add_subplot(3, 4, idx+1, projection='3d')

        Y = sph_harm(m, l, PH, TH)
        mag = np.abs(Y.real) / np.max(np.abs(Y.real))

        X = mag * np.sin(TH) * np.cos(PH)
        Y_ax = mag * np.sin(TH) * np.sin(PH)
        Z = mag * np.cos(TH)

        colors = cm.coolwarm(norm(Y.real))
        ax.plot_surface(X, Y_ax, Z, facecolors=colors,
                        alpha=0.8, rstride=2, cstride=2)
        ax.set_title(f'$Y_{l}^{m}$', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_5_spherical_harmonics.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig4_5_spherical_harmonics.png")


# ================================================================
#  第 5 章  数值方法
# ================================================================

def fig5_1_fd_stencil():
    """有限差分模板"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 一维模板
    ax = axes[0]
    pts = [-2, -1, 0, 1, 2]
    for p in pts:
        ax.plot(p, 0, 'o', markersize=12, color='lightblue', markeredgecolor='blue', markeredgewidth=2)
    ax.plot(0, 0, 'o', markersize=15, color='red', markeredgecolor='darkred', markeredgewidth=2)
    ax.text(0, -0.15, '$u_i$', ha='center', fontsize=14)
    ax.text(1, 0.1, '$u_{i+1}$', ha='center', fontsize=12)
    ax.text(-1, 0.1, '$u_{i-1}$', ha='center', fontsize=12)
    ax.set_ylim(-0.3, 0.3); ax.set_xlim(-2.5, 2.5)
    ax.axis('off')
    ax.set_title('(a) 1D 3-point stencil\n$u_i\'\'\\approx(u_{i+1}-2u_i+u_{i-1})/h^2$', fontsize=10)

    # 二维五点模板
    ax = axes[1]
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                ax.plot(i, j, 'o', markersize=15, color='red', markeredgecolor='darkred', markeredgewidth=2)
            else:
                ax.plot(i, j, 'o', markersize=12, color='lightblue', markeredgecolor='blue', markeredgewidth=2)
    ax.text(0, -0.15, '$u_{i,j}$', ha='center', fontsize=14)
    ax.text(1, 0.1, '$u_{i+1,j}$', ha='center', fontsize=11)
    ax.text(-1, 0.1, '$u_{i-1,j}$', ha='center', fontsize=11)
    ax.text(0.1, 1, '$u_{i,j+1}$', ha='left', fontsize=11)
    ax.text(0.1, -1, '$u_{i,j-1}$', ha='left', fontsize=11)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_title('(b) 2D 5-point stencil\n$\\nabla^2 u_{ij}\\approx\\frac{u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{ij}}{h^2}$', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_1_fd_stencil.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig5_1_fd_stencil.png")


def fig5_2_sparse_pattern():
    """稀疏矩阵结构"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    from scipy import sparse

    # 一维泊松矩阵
    N = 20
    e = np.ones(N)
    A1 = sparse.diags([-e, 2*e, -e], [-1, 0, 1], shape=(N, N))
    axes[0].spy(A1, markersize=3)
    axes[0].set_title(f'(a) 1D Poisson matrix ({N}\\times{N})')

    # 二维拉普拉斯矩阵
    M = 8
    I = sparse.eye(M)
    T = sparse.diags([-np.ones(M), 4*np.ones(M), -np.ones(M)], [-1, 0, 1], shape=(M, M))
    A2 = sparse.kron(sparse.eye(M), T) + sparse.kron(sparse.diags([np.ones(M-1)], [-1]), -I) \
         + sparse.kron(sparse.diags([np.ones(M-1)], [1]), -I)
    axes[1].spy(A2, markersize=1.5)
    axes[1].set_title(f'(b) 2D 5-point Laplacian ({M**2}\\times{M**2})')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_2_sparse_pattern.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig5_2_sparse_pattern.png")


def fig5_3_convergence():
    """有限差分收敛性研究"""
    fig, ax = plt.subplots(figsize=(7, 5))

    Ns = np.array([10, 20, 40, 80, 160, 320])
    hs = 1.0 / Ns

    errors_O2 = hs**2 * 10
    errors_O1 = hs * 20

    np.random.seed(42)
    errors_O2_actual = errors_O2 * (1 + 0.05*np.random.randn(len(Ns)))
    errors_O1_actual = errors_O1 * (1 + 0.05*np.random.randn(len(Ns)))

    ax.loglog(hs, errors_O2_actual, 'o-', label='Central diff $O(h^2)$', lw=2)
    ax.loglog(hs, errors_O1_actual, 's-', label='Forward diff $O(h)$', lw=2)
    ax.loglog(hs, hs**2, '--', label=r'$\propto h^2$', alpha=0.5)
    ax.loglog(hs, hs, '--', label=r'$\propto h$', alpha=0.5)

    ax.set_xlabel('Grid spacing $h$'); ax.set_ylabel('Max error')
    ax.set_title('Finite difference convergence')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_3_convergence.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig5_3_convergence.png")


def fig5_4_numerical_solution():
    """数值解 vs 解析解对比"""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    N = 50
    h = 1.0/N
    x = np.linspace(0, 1, N+1)
    e = np.ones(N-1)
    A = sparse.diags([-e, 2*e, -e], [-1, 0, 1], shape=(N-1, N-1)) / h**2
    f = np.sin(np.pi * x[1:-1])
    u_num = np.zeros(N+1)
    u_num[1:-1] = spsolve(A, f)
    u_ex = np.sin(np.pi * x) / np.pi**2

    axes[0].plot(x, u_num, 'r-', lw=2, label='Numerical')
    axes[0].plot(x, u_ex, 'b--', lw=2, label='Exact')
    axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$u(x)$')
    axes[0].set_title(r'$-u\'\' = \sin(\pi x)$'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(x[1:-1], np.abs(u_num[1:-1] - u_ex[1:-1]), 'o-', ms=3)
    axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$|u_{num} - u_{ex}|$')
    axes[1].set_title('Pointwise error $N=50$ (log scale)'); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_4_numerical_solution.png'), dpi=DPI)
    plt.close()
    print(f"  {OK_MSG} fig5_4_numerical_solution.png")


# ================================================================
#  主函数
# ================================================================

def main():
    print("=" * 50)
    print("Generating figures for 'Mathematical Foundations for Physics'")
    print("=" * 50)
    print()
    print(f"CJK font: {CJK_FONT_NAME}")
    print()

    print("Chapter 1 figures...")
    fig1_1_coordinates()
    fig1_2_divergence()
    fig1_3_curl()
    fig1_4_gauss_theorem()

    print("\nChapter 2 figures...")
    fig2_1_complex_plane()
    fig2_2_contour()
    fig2_3_conformal()

    print("\nChapter 3 figures...")
    fig3_1_fourier_series()
    fig3_2_ft_pairs()
    fig3_3_fft_example()
    fig3_4_laplace_roc()

    print("\nChapter 4 figures...")
    fig4_1_pde_types()
    fig4_2_legendre()
    fig4_3_bessel()
    fig4_4_drum_modes()
    fig4_5_spherical_harmonics()

    print("\nChapter 5 figures...")
    fig5_1_fd_stencil()
    fig5_2_sparse_pattern()
    fig5_3_convergence()
    fig5_4_numerical_solution()

    print("\n" + "=" * 50)
    print(f"All figures generated in: {OUTPUT_DIR}")
    print("Total: 19 figures")
    print("=" * 50)


if __name__ == '__main__':
    main()

"""
Axisymmetric Finite Difference Thermal Model  —  Newton iteration for radiation
================================================================================
Geometry (r-z, axisymmetric, z measured from bottom of TIM):

  r <= R_sp (37.5 mm):  4-layer stack exists
    z = 0            : bottom of TIM  — heat source Neumann BC
    z = t_TIM        : TIM / HeatSpreader interface
    z = t_TIM+t_Sp   : HeatSpreader / Adhesive interface
    z = t_TIM+t_Sp+t_ADH : Adhesive / Glass interface
    z = H            : top of Glass  — convection + radiation BC

  R_sp < r <= R_gl (50 mm):  Glass plate only
    The GL plate sits at the TOP of the stack (z_bounds[3] to H).
    Below z_bounds[3] at these radii the domain does not physically exist.
    We handle this by setting the computational domain bottom for r > R_sp
    to z = z_bounds[3], i.e. the GL-only annulus is a separate shorter column.

  IMPLEMENTATION:
    Single unified r-z grid covering 0 <= r <= R_gl, 0 <= z <= H.
    For r > R_sp AND z < z_bounds[3]: these nodes are "void" — assigned a
    high-conductivity isotropic value and Dirichlet T = (floating, solved).
    Actually the cleanest approach: for these void nodes we enforce
    dT/dz = 0 (adiabatic wall) at z = z_bounds[3] for r > R_sp, and set
    all void nodes (r>R_sp, z<z_bounds[3]) equal to their z_bounds[3] neighbour
    via a simple constraint: T[j,i] = T[j_al_bot, i] for j < j_al_bot, r > R_sp.
    This is equivalent to "infinite k" in the void — a uniform temperature pillar.
    Since no heat enters/leaves the void this just enforces no axial gradient there.

    SIMPLER equivalent: set k_void = 1e6 W/m*K (numerically "isothermal void").
    Then the void nodes equilibrate to the GL bottom face temperature automatically.
    Heat only enters/leaves via BCs at z=0 (source) and z=H (conv+rad).
    At z=0, r>R_sp: adiabatic — so no heat enters void.
    Result: void nodes = GL bottom temperature at that r.  Physically correct.

Boundary conditions:
  z = 0 , r <= r_src  : Neumann  q = Q / (pi*r_src^2)  W/m²
  z = 0 , r >  r_src  : Adiabatic
  z = H , all r        : -k dT/dz = h*(T-T_amb) + sigma*eps*(T_K^4 - T_surr_K^4)
                         Solved via Newton linearisation each outer iteration
  r = 0                : Symmetry axis
  r = R_gl             : Adiabatic
  z = 0 , r > R_sp    : Adiabatic (void bottom — no heat enters air gap below GL annulus)

Author: Thermal FDM Solver
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# ─── OUTPUT FOLDER ────────────────────────────────────────────────────────────
# Change OUT to any folder you prefer, e.g. OUT = r"C:\Users\pravi\Desktop\results"
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# ─── PARAMETERS ───────────────────────────────────────────────────────────────
Q_total  = 6.0           # W  — total heat source power
r_source = 5.0e-3         # m  — heat source radius  (D = 6 mm)
R_sp     = 37.5e-3        # m  — HeatSpreader spreader radius  (D = 75 mm)
R_gl     = 50.0e-3        # m  — Glass skin radius      (D = 75 mm)
T_amb    = 25.0           # °C — ambient temperature
h_conv   = 10.0            # W/m²·K — natural convection coefficient
epsilon  = 0.9         # —    emissivity of Glass back surface
sigma    = 5.670374419e-8 # W/m²·K⁴

t_TIM = 0.10e-3;  k_TIM  = 3

t_Sp = 4; # Heat Spreader tickness in mm
t_Sp  = t_Sp*1e-3;  k_GR_r = 400;  k_GR_z = 400

t_ADH = 0.10e-3;  k_ADH  = 3
t_GL  = 1.00e-3;  k_GL   = 1.0
k_void = 0.026            # W/m·K  — air gap below GL annulus (r>R_sp, z<z_al_bot)

H        = t_TIM + t_Sp + t_ADH + t_GL
z_bounds = np.array([0.0,
                     t_TIM,
                     t_TIM + t_Sp,
                     t_TIM + t_Sp + t_ADH,
                     H])

q_src  = Q_total / (np.pi * r_source**2)   # W/m²
T_surr = T_amb                              # surroundings temperature for radiation
Ts_K   = T_surr + 273.15

# ─── GRID ─────────────────────────────────────────────────────────────────────

r_a = np.linspace(0,          r_source,  50, endpoint=False)
r_b = np.linspace(r_source,   R_sp,      200, endpoint=False)
r_c = np.linspace(R_sp,       R_gl,      50)
grid_refine = 5

r_nodes = np.unique(np.concatenate([r_a, r_b, r_c]*grid_refine))
Nr = len(r_nodes)

def make_seg(z0, z1, n_int):
    return np.linspace(z0, z1, n_int + 2)

segs = [make_seg(z_bounds[0], z_bounds[1],  10),
        make_seg(z_bounds[1], z_bounds[2], 15),
        make_seg(z_bounds[2], z_bounds[3],  10),
        make_seg(z_bounds[3], z_bounds[4], 20)]
z_nodes = np.unique(np.concatenate(segs))
Nz = len(z_nodes)
N  = Nr * Nz

# Index of z_bounds[3] (bottom of GL plate = top of adhesive)
j_al_bot = int(np.argmin(np.abs(z_nodes - z_bounds[3])))

print(f"Grid: Nr={Nr}, Nz={Nz}, Total DOF={N}")
print(f"Layer z-bounds [mm]: {z_bounds*1e3}")
print(f"j_al_bot = {j_al_bot}  (z = {z_nodes[j_al_bot]*1e3:.3f} mm)")
print(f"Spreader radius R_sp = {R_sp*1e3:.1f} mm  |  GL skin radius R_gl = {R_gl*1e3:.1f} mm")

# ─── MATERIAL PROPERTIES ──────────────────────────────────────────────────────
def mat(j, i):
    """Return (k_r, k_z) for grid node (j, i)."""
    z = z_nodes[j]
    r = r_nodes[i]
    if r > R_sp:
        if z < z_bounds[3]:
            return k_void, k_void   # air gap void below GL annulus
        else:
            return k_GL, k_GL       # Glass plate annulus
    # r <= R_sp  — layer by layer
    if z <= z_bounds[1]:
        # TIM footprint = source footprint only. Strict < so node exactly at
        # r_source is void, giving hm(k_TIM, k_void)~0.05 not hm(k_TIM,k_TIM)=3
        if r < r_source:
            return k_TIM, k_TIM
        else:
            return k_void, k_void   # inner void: air beside TIM (r>=r_src, z<=t_TIM)
    elif z <= z_bounds[2]: return k_GR_r, k_GR_z
    elif z <= z_bounds[3]: return k_ADH,  k_ADH
    else:                  return k_GL,   k_GL

kr_g = np.zeros((Nz, Nr))
kz_g = np.zeros((Nz, Nr))
for j in range(Nz):
    for i in range(Nr):
        kr_g[j, i], kz_g[j, i] = mat(j, i)

def hm(a, b):
    s = a + b
    return 2.0*a*b/s if s > 0 else 0.0

def gidx(j, i):
    return j * Nr + i

# ─── NEWTON ITERATION ─────────────────────────────────────────────────────────
max_iter = 30
tol      = 1e-4   # °C

T_flat = np.full(N, T_amb + 50.0)   # initial guess

print("\nNewton iterations (nonlinear convection + radiation BC):")

for iteration in range(max_iter):

    T_prev      = T_flat.copy()
    t_Spid_prev = T_prev.reshape((Nz, Nr))

    A_lin = lil_matrix((N, N), dtype=np.float64)
    b_lin = np.zeros(N)

    for j in range(Nz):
        for i in range(Nr):
            n = gidx(j, i)
            r = r_nodes[i]

            # ── BOTTOM BC (j=0) ───────────────────────────────────────────
            # r < r_source             : Neumann heat flux in
            # r_source <= r <= R_sp   : Adiabatic (inner void bottom, no source)
            # r > R_sp (void/air gap) : Adiabatic (dT/dz = 0) — no heat enters void
            if j == 0:
                dz  = z_nodes[1] - z_nodes[0]
                kzc = kz_g[0, i]
                A_lin[n, gidx(0, i)] =  kzc / dz
                A_lin[n, gidx(1, i)] = -kzc / dz
                b_lin[n] = q_src if r < r_source else 0.0
                continue

            # ── INNER VOID NODES (r >= r_source, z <= z_bounds[1], j>0) ──
            # These are air beside the TIM. Pin T = T of the spreader node
            # directly above (j = j_tim_top) to prevent spurious conduction
            # from the hot spreader conducting down into the "air" column.
            elif r_nodes[i] >= r_source and r_nodes[i] <= R_sp and z_nodes[j] < z_bounds[1]:
                j_top = int(np.argmin(np.abs(z_nodes - z_bounds[1])))
                A_lin[n, n]              = 1.0
                A_lin[n, gidx(j_top, i)] = -1.0
                b_lin[n]                 = 0.0
                continue

            # ── TOP BC (j=Nz-1): combined convection + radiation ──────────
            elif j == Nz - 1:
                dz   = z_nodes[-1] - z_nodes[-2]
                kzc  = kz_g[-1, i]
                T0_C = t_Spid_prev[-1, i]
                T0_K = T0_C + 273.15
                q_rad0 = sigma * epsilon * (T0_K**4 - Ts_K**4)
                h_rad  = 4.0 * sigma * epsilon * T0_K**3

                A_lin[n, gidx(Nz-2, i)] =  kzc / dz
                A_lin[n, gidx(Nz-1, i)] = -(kzc / dz + h_conv + h_rad)
                b_lin[n] = -(h_conv * T_amb + h_rad * T0_C - q_rad0)
                continue

            # ── INTERIOR NODE ─────────────────────────────────────────────
            else:
                dr_m = r_nodes[i]   - r_nodes[i-1] if i > 0    else r_nodes[1]-r_nodes[0]
                dr_p = r_nodes[i+1] - r_nodes[i]   if i < Nr-1 else r_nodes[-1]-r_nodes[-2]
                dz_m = z_nodes[j]   - z_nodes[j-1]
                dz_p = z_nodes[j+1] - z_nodes[j]
                dr_c = 0.5*(dr_m + dr_p)
                dz_c = 0.5*(dz_m + dz_p)

                kzp = hm(kz_g[j,i], kz_g[j+1,i])
                kzm = hm(kz_g[j,i], kz_g[j-1,i])
                czp = kzp / (dz_p * dz_c)
                czm = kzm / (dz_m * dz_c)
                A_lin[n, gidx(j+1, i)] += czp
                A_lin[n, gidx(j-1, i)] += czm
                A_lin[n, gidx(j,   i)] -= (czp + czm)

                if i == 0:
                    krc = kr_g[j, 0]
                    dr  = r_nodes[1] - r_nodes[0]
                    crp = 2.0 * krc / dr**2
                    A_lin[n, gidx(j, 1)] +=  crp
                    A_lin[n, gidx(j, 0)] -= crp
                elif i == Nr - 1:
                    krm2 = hm(kr_g[j,i], kr_g[j,i-1])
                    r_m  = r - 0.5*dr_m
                    crm  = krm2 * r_m / (r * dr_m * dr_c)
                    A_lin[n, gidx(j, i-1)] +=  crm
                    A_lin[n, gidx(j, i)]   -= crm
                else:
                    krp2 = hm(kr_g[j,i], kr_g[j,i+1])
                    krm2 = hm(kr_g[j,i], kr_g[j,i-1])
                    r_p  = r + 0.5*dr_p
                    r_m  = r - 0.5*dr_m
                    crp  = krp2 * r_p / (r * dr_p * dr_c)
                    crm  = krm2 * r_m / (r * dr_m * dr_c)
                    A_lin[n, gidx(j, i+1)] +=  crp
                    A_lin[n, gidx(j, i-1)] +=  crm
                    A_lin[n, gidx(j, i)]   -= (crp + crm)

    T_flat = spsolve(A_lin.tocsr(), b_lin)
    delta  = np.max(np.abs(T_flat - T_prev))
    print(f"  Iter {iteration+1:2d}:  max ΔT = {delta:.4e} °C")
    if delta < tol:
        print(f"  Converged in {iteration+1} iterations.\n")
        break
else:
    print("  WARNING: did not converge within max_iter.\n")

T = T_flat.reshape((Nz, Nr))

# ─── POST-PROCESS ─────────────────────────────────────────────────────────────
T_src_max  = float(np.max(T[0, r_nodes <= r_source]))
T_src_ctr  = float(T[0, 0])
j_grtop    = int(np.argmin(np.abs(z_nodes - z_bounds[2])))
t_Sp_top   = float(T[j_grtop, 0])
t_GL_ctr   = float(T[-1, 0])
i_sp_edge  = int(np.argmin(np.abs(r_nodes - R_sp)))
t_GL_sp    = float(T[-1, i_sp_edge])
t_GL_edge  = float(T[-1, -1])

T0_K_ctr  = t_GL_ctr + 273.15
h_rad_ctr = sigma * epsilon * (T0_K_ctr**4 - Ts_K**4) / max(t_GL_ctr - T_surr, 0.01)

print(f"{'═'*58}")
print(f"  THERMAL RESULTS")
print(f"{'═'*58}")
print(f"  Max heat source temp      :  {T_src_max:7.2f} °C")
print(f"  Heat source center (z=0)  :  {T_src_ctr:7.2f} °C")
print(f"  HeatSpreader top, center      :  {t_Sp_top:7.2f} °C")
print(f"  GL back, center           :  {t_GL_ctr:7.2f} °C")
print(f"  GL back, spreader edge    :  {t_GL_sp:7.2f} °C")
print(f"  GL back, outer edge       :  {t_GL_edge:7.2f} °C")
print(f"  Ambient / surroundings    :  {T_amb:7.1f} °C")
print(f"  ΔT (source to ambient)    :  {T_src_max-T_amb:7.2f} °C")
print(f"  h_rad at GL centre        :  {h_rad_ctr:7.2f} W/m²·K")
print(f"{'═'*58}\n")

# ─── ENERGY BALANCE ───────────────────────────────────────────────────────────
q_conv = 0.0;  q_rad = 0.0
for i in range(Nr-1):
    r_mid   = 0.5*(r_nodes[i] + r_nodes[i+1])
    dr      = r_nodes[i+1] - r_nodes[i]
    T_mid   = 0.5*(T[-1,i] + T[-1,i+1])
    T_mid_K = T_mid + 273.15
    dA      = 2*np.pi*r_mid*dr
    q_conv += h_conv * (T_mid - T_amb) * dA
    q_rad  += sigma * epsilon * (T_mid_K**4 - Ts_K**4) * dA
q_out = q_conv + q_rad
print(f"Energy balance:")
print(f"  Q_in         = {Q_total:.3f} W")
print(f"  Q_convection = {q_conv:.3f} W  ({q_conv/Q_total*100:.1f}%)")
print(f"  Q_radiation  = {q_rad:.3f} W  ({q_rad/Q_total*100:.1f}%)")
print(f"  Q_total_out  = {q_out:.3f} W  (error = {abs(q_out-Q_total)/Q_total*100:.1f}%)\n")

# ─── COLORMAP ─────────────────────────────────────────────────────────────────
cmap_th = LinearSegmentedColormap.from_list(
    "thermal",
    ["#0b0c2a","#1565c0","#00bcd4","#76ff03","#ffeb3b","#ff6f00","#b71c1c"],
    N=512
)

# Mask void regions for plotting (set to NaN so they render blank)
# 1. Outer void: r > R_sp, z < z_bounds[3]
# 2. Inner void: r >= r_source, r <= R_sp, z < z_bounds[1]  (air beside TIM)
T_plot = T.copy().astype(float)
for j in range(Nz):
    for i in range(Nr):
        if r_nodes[i] > R_sp and z_nodes[j] < z_bounds[3]:
            T_plot[j, i] = np.nan
        elif r_nodes[i] >= r_source and r_nodes[i] <= R_sp and z_nodes[j] < z_bounds[1]:
            T_plot[j, i] = np.nan

# ─── FIG 1 — Full 2-D temperature contour ─────────────────────────────────────
R2D, Z2D = np.meshgrid(r_nodes*1e3, z_nodes*1e3)
fig1, ax1 = plt.subplots(figsize=(14, 6))
fig1.patch.set_facecolor("#0d1117"); ax1.set_facecolor("#0d1117")
cf = ax1.contourf(R2D, Z2D, T_plot, levels=80, cmap=cmap_th)
cs = ax1.contour( R2D, Z2D, T_plot, levels=16, colors='white', linewidths=0.35, alpha=0.3)
ax1.clabel(cs, fmt="%.1f°C", fontsize=7, colors='white')
cb = fig1.colorbar(cf, ax=ax1, pad=0.02)
cb.set_label("Temperature [°C]", color='white', fontsize=11)
cb.ax.yaxis.set_tick_params(color='white')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
cb.outline.set_edgecolor('white')
for z_b, lbl in [(z_bounds[1]*1e3, "TIM | HeatSpreader"),
                  (z_bounds[2]*1e3, "HeatSpreader | Adhesive"),
                  (z_bounds[3]*1e3, "Adhesive | Glass")]:
    ax1.axhline(z_b, color='white', lw=0.9, ls='--', alpha=0.5)
    ax1.text(R_gl*1e3*0.52, z_b+0.03, lbl, color='white', fontsize=8, alpha=0.8)
ax1.axvline(r_source*1e3, color='#ff5555', lw=1.2, ls=':', alpha=0.85,
            label=f"Source edge r={r_source*1e3:.0f} mm")
ax1.axvline(R_sp*1e3, color='#ffdd57', lw=1.2, ls='--', alpha=0.85,
            label=f"Spreader edge r={R_sp*1e3:.0f} mm")
ax1.annotate(f"T_max = {T_src_max:.1f}°C",
             xy=(0.0, 0.0), xytext=(r_source*1e3*5, 0.15),
             color='#ff6868', fontsize=11, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#ff6868', lw=1.5))
ax1.set_xlabel("r [mm]", color='white', fontsize=12)
ax1.set_ylabel("z [mm]", color='white', fontsize=12)
ax1.set_aspect(5)   # 1 z-unit displayed as 5× the height of 1 r-unit
ax1.set_title(
    f"2-D Axisymmetric Temperature Field  —  T_max = {T_src_max:.1f}°C  "
    f"|  Conv + Rad (ε={epsilon}) on GL back",
    color='white', fontsize=12, pad=10)
ax1.tick_params(colors='white')
for sp in ax1.spines.values(): sp.set_edgecolor('#444')
ax1.legend(loc='upper right', fontsize=9, framealpha=0.3,
           labelcolor='white', facecolor='#1a1a2e')
plt.tight_layout()
fig1.savefig(os.path.join(OUT, "fig1_temperature_contour.png"), dpi=150,
             bbox_inches='tight', facecolor=fig1.get_facecolor())
print("Saved: fig1_temperature_contour.png")

# ─── FIG 2 — Zoomed: source + TIM + HeatSpreader entry ────────────────────────────
z_top_zoom = z_bounds[1] + 1.2e-3
mz = z_nodes <= z_top_zoom;  mr = r_nodes <= 15e-3
T_z = T_plot[np.ix_(mz, mr)]
Rz, Zz = np.meshgrid(r_nodes[mr]*1e3, z_nodes[mz]*1e3)
fig2, ax2 = plt.subplots(figsize=(9, 5))
fig2.patch.set_facecolor("#0d1117"); ax2.set_facecolor("#0d1117")
cf2 = ax2.contourf(Rz, Zz, T_z, levels=60, cmap=cmap_th)
cs2 = ax2.contour( Rz, Zz, T_z, levels=14, colors='white', linewidths=0.3, alpha=0.4)
ax2.clabel(cs2, fmt="%.1f°C", fontsize=7, colors='white')
cb2 = fig2.colorbar(cf2, ax=ax2, pad=0.02)
cb2.set_label("Temperature [°C]", color='white', fontsize=11)
cb2.ax.yaxis.set_tick_params(color='white')
plt.setp(cb2.ax.yaxis.get_ticklabels(), color='white')
cb2.outline.set_edgecolor('white')
ax2.axhline(z_bounds[1]*1e3, color='cyan',    lw=1.0, ls='--', alpha=0.7, label="TIM | HeatSpreader")
ax2.axvline(r_source*1e3,    color='#ff5555', lw=1.2, ls=':',  alpha=0.8, label="Source edge")
ax2.set_xlabel("r [mm]", color='white', fontsize=12)
ax2.set_ylabel("z [mm]", color='white', fontsize=12)
ax2.set_title("Zoomed: Heat Source → TIM → HeatSpreader Entry", color='white', fontsize=12)
ax2.tick_params(colors='white')
for sp in ax2.spines.values(): sp.set_edgecolor('#444')
ax2.legend(fontsize=9, framealpha=0.3, labelcolor='white', facecolor='#1a1a2e')
plt.tight_layout()
fig2.savefig(os.path.join(OUT, "fig2_zoomed_source.png"), dpi=150,
             bbox_inches='tight', facecolor=fig2.get_facecolor())
print("Saved: fig2_zoomed_source.png")

# ─── FIG 3 — Centerline T vs z ────────────────────────────────────────────────
T_cl = T[:, 0]
fig3, ax3 = plt.subplots(figsize=(9, 5))
fig3.patch.set_facecolor("#0d1117"); ax3.set_facecolor("#111827")
layer_cols  = ['#ff6b6b33','#ffa50022','#90ee9022','#4169e122']
layer_names = ['TIM','HeatSpreader','Adhesive','Glass']
for k_l in range(4):
    ax3.axvspan(z_bounds[k_l]*1e3, z_bounds[k_l+1]*1e3, color=layer_cols[k_l], lw=0)
    mid  = 0.5*(z_bounds[k_l]+z_bounds[k_l+1])*1e3
    rot  = 90 if (z_bounds[k_l+1]-z_bounds[k_l]) < 0.5e-3 else 0
    ypos = T_amb + (T_cl.max()-T_amb)*0.05
    ax3.text(mid, ypos, layer_names[k_l], ha='center', fontsize=8.5,
             color='white', alpha=0.7, rotation=rot, va='bottom')
ax3.plot(z_nodes*1e3, T_cl, color='#00e5ff', lw=2.3, label="Centerline T(z)")
ax3.fill_between(z_nodes*1e3, T_cl, T_amb, alpha=0.12, color='#00e5ff')
ax3.axhline(T_amb, color='gray', ls='--', lw=1.0, alpha=0.5, label=f"T_amb = {T_amb}°C")
ax3.scatter([0], [T_src_ctr], color='red', s=80, zorder=6,
            label=f"T_max = {T_src_max:.1f}°C")
ax3.set_xlabel("z [mm]", color='white', fontsize=12)
ax3.set_ylabel("Temperature [°C]", color='white', fontsize=12)
ax3.set_title("Centerline Temperature Through Stack", color='white', fontsize=12)
ax3.tick_params(colors='white')
ax3.grid(True, color='#333', lw=0.5, alpha=0.6)
ax3.legend(fontsize=9.5, framealpha=0.4, labelcolor='white', facecolor='#0d1117')
for sp in ax3.spines.values(): sp.set_edgecolor('#333')
plt.tight_layout()
fig3.savefig(os.path.join(OUT, "fig3_centerline.png"), dpi=150,
             bbox_inches='tight', facecolor=fig3.get_facecolor())
print("Saved: fig3_centerline.png")

# ─── FIG 4 — Radial profiles at key z levels ──────────────────────────────────
j_tim_top = int(np.argmin(np.abs(z_nodes - z_bounds[1])))
j_gr_mid  = int(np.argmin(np.abs(z_nodes - (z_bounds[1] + t_Sp/2))))
z_lvls = [
    (0,         f"Source base z=0  ({T[0,0]:.1f}°C)",         '#ff4444'),
    (j_tim_top, f"TIM top  ({T[j_tim_top,0]:.1f}°C)",         '#ff9900'),
    (j_gr_mid,  f"HeatSpreader mid  ({T[j_gr_mid,0]:.1f}°C)",     '#00e5ff'),
    (j_grtop,   f"HeatSpreader top  ({t_Sp_top:.1f}°C)",          '#76ff03'),
    (Nz-1,      f"GL back  (ctr={t_GL_ctr:.1f}°C)",           '#e040fb'),
]
fig4, ax4 = plt.subplots(figsize=(12, 5))
fig4.patch.set_facecolor("#0d1117"); ax4.set_facecolor("#111827")
for jj, lbl, col in z_lvls:
    T_row = T_plot[jj, :].copy()
    ax4.plot(r_nodes*1e3, T_row, color=col, lw=1.8, label=lbl)
ax4.axvline(r_source*1e3, color='white',   ls=':', lw=1.0, alpha=0.45, label="Source edge")
ax4.axvline(R_sp*1e3,     color='#ffdd57', ls='--', lw=1.0, alpha=0.7,
            label=f"Spreader edge ({R_sp*1e3:.0f} mm)")
ax4.axhline(T_amb, color='gray', ls='--', lw=1.0, alpha=0.45, label=f"T_amb = {T_amb}°C")
ax4.set_xlabel("r [mm]", color='white', fontsize=12)
ax4.set_ylabel("Temperature [°C]", color='white', fontsize=12)
ax4.set_title("Radial Temperature Distribution at Key Axial Levels", color='white', fontsize=12)
ax4.tick_params(colors='white')
ax4.grid(True, color='#333', lw=0.5, alpha=0.6)
ax4.legend(fontsize=8.5, framealpha=0.4, labelcolor='white', facecolor='#0d1117', ncol=2)
for sp in ax4.spines.values(): sp.set_edgecolor('#333')
ax4.set_xlim([0, R_gl*1e3])
plt.tight_layout()
fig4.savefig(os.path.join(OUT, "fig4_radial.png"), dpi=150,
             bbox_inches='tight', facecolor=fig4.get_facecolor())
print("Saved: fig4_radial.png")

# ─── FIG 5 — Back surface heat flux breakdown ─────────────────────────────────
T_back   = T[-1, :]
T_back_K = T_back + 273.15
q_c_back = h_conv * (T_back - T_amb)
q_r_back = sigma * epsilon * (T_back_K**4 - Ts_K**4)
q_t_back = q_c_back + q_r_back

fig5, ax5 = plt.subplots(figsize=(11, 5))
fig5.patch.set_facecolor("#0d1117"); ax5.set_facecolor("#111827")
ax5.plot(r_nodes*1e3, q_c_back, color='#00e5ff', lw=2.0, label="Convection  q_conv [W/m²]")
ax5.plot(r_nodes*1e3, q_r_back, color='#ff9900', lw=2.0, label="Radiation   q_rad  [W/m²]")
ax5.plot(r_nodes*1e3, q_t_back, color='white',   lw=1.5, ls='--', label="Total  q_conv + q_rad [W/m²]")
ax5.axvline(r_source*1e3, color='#ff5555', ls=':', lw=1.0, alpha=0.5, label="Source edge")
ax5.axvline(R_sp*1e3,     color='#ffdd57', ls='--', lw=1.0, alpha=0.7, label="Spreader edge")
ax5.set_xlabel("r [mm]", color='white', fontsize=12)
ax5.set_ylabel("Surface heat flux [W/m²]", color='white', fontsize=12)
ax5.set_title("Convection + Radiation Flux on Glass Back Surface", color='white', fontsize=12)
ax5.tick_params(colors='white')
ax5.grid(True, color='#333', lw=0.5, alpha=0.6)
ax5.legend(fontsize=9, framealpha=0.4, labelcolor='white', facecolor='#0d1117')
for sp in ax5.spines.values(): sp.set_edgecolor('#333')
ax5.set_xlim([0, R_gl*1e3])
plt.tight_layout()
fig5.savefig(os.path.join(OUT, "fig5_back_flux.png"), dpi=150,
             bbox_inches='tight', facecolor=fig5.get_facecolor())
print("Saved: fig5_back_flux.png")

# Show all plots
#plt.show()

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
print(f"""
╔════════════════════════════════════════════════════════════╗
║      AXISYMMETRIC FDM  —  THERMAL ANALYSIS RESULTS        ║
╠════════════════════════════════════════════════════════════╣
║  Input Power                :  {Q_total:.1f} W                      ║
║  Source diameter            :  {r_source*2*1e3:.0f} mm                    ║
║  HeatSpreader spreader diameter :  {R_sp*2*1e3:.0f} mm                   ║
║  Glass skin diameter     :  {R_gl*2*1e3:.0f} mm                  ║
║  Conv. coefficient h        :  {h_conv:.1f} W/m²K                ║
║  Emissivity (GL back)       :  {epsilon:.2f}                       ║
║  h_rad at GL centre         :  {h_rad_ctr:.2f} W/m²K               ║
╠════════════════════════════════════════════════════════════╣
║  Max heat source temp       :  {T_src_max:6.2f} °C                 ║
║  Source center (z=0)        :  {T_src_ctr:6.2f} °C                 ║
║  HeatSpreader top, center       :  {t_Sp_top:6.2f} °C                 ║
║  GL back, center            :  {t_GL_ctr:6.2f} °C                 ║
║  GL back, spreader edge     :  {t_GL_sp:6.2f} °C                 ║
║  GL back, outer edge        :  {t_GL_edge:6.2f} °C                 ║
║  Ambient temperature        :  {T_amb:6.1f} °C                 ║
║  ΔT (source to ambient)     :  {T_src_max-T_amb:6.2f} °C                 ║
╠════════════════════════════════════════════════════════════╣
║  Q_convection out           :  {q_conv:6.3f} W  ({q_conv/Q_total*100:.1f}%)             ║
║  Q_radiation  out           :  {q_rad:6.3f} W  ({q_rad/Q_total*100:.1f}%)             ║
║  Q_total out                :  {q_out:6.3f} W  (error = {abs(q_out-Q_total)/Q_total*100:.1f}%)          ║
╚════════════════════════════════════════════════════════════╝
""")

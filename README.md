# Graphite vs Copper Heat Spreader — Axisymmetric FDM Thermal Study

A 2D axisymmetric finite difference model (FDM) comparing graphite and copper heat
spreaders across a thickness sweep of 0.01–5 mm. Solved via steady-state conduction
with natural convection and radiation boundary conditions on the glass back surface.
Results cross-checked against Siemens FloEFD thermal simulation at three thickness points.

---

## Live Site

Deployed via GitHub Pages — `index.html` is the study page.

> **Deploy:** Settings → Pages → Source: Deploy from branch → `main` → `/` (root)  
> Site URL: `https://<username>.github.io/<repo-name>/`

---

## Repository Files

| File | Description |
|------|-------------|
| `index.html` | GitHub Pages study site — executive summary, thermal stack schematic, parametric chart, validation table |
| `thermal_fdm_axisymmetric_Gr.py` | Python FDM solver — Graphite spreader (anisotropic k) |
| `thermal_fdm_axisymmetric_Cu.py` | Python FDM solver — Copper spreader (isotropic k) |
| `Heat_Spreader.xlsx` | Parametric study data — FDM results + FloEFD cross-check points (Sheet 1 = current) |
| `_config.yml` | GitHub Pages config (disables Jekyll theme) |
| `README.md` | This file |

---

## Quick Start

```bash
pip install numpy scipy matplotlib
python thermal_fdm_axisymmetric_Gr.py   # graphite spreader study
python thermal_fdm_axisymmetric_Cu.py   # copper spreader study
```

Each script saves five figures to its own directory:
`fig1_temperature_contour.png`, `fig2_zoomed_source.png`, `fig3_centerline.png`,
`fig4_radial.png`, `fig5_back_flux.png`

---

## Problem Definition

### Geometry

2D axisymmetric (r-z), single unified grid:

```
r = 0 ─────── r_src = 5 mm ──────── R_sp = 37.5 mm ──── R_gl = 50 mm
│                   │                       │                   │
│  TIM + Spreader   │  Spreader + Adh +     │  Glass only       │
│  + Adh + Glass    │  Glass  (no TIM)      │  (annulus)        │
│                   │                       │                   │
z=0: q_src          z=0: adiabatic          z=0: adiabatic (void)
```

TIM footprint equals source footprint only (r < 5 mm). The region
r ≥ 5 mm at the TIM z-level is air (inner void, k = 0.026 W/m·K).

### Material Stack (along centerline)

| Layer | k_r [W/m·K] | k_z [W/m·K] | Thickness |
|-------|-------------|-------------|-----------|
| Glass | 1.0 | 1.0 | 1.00 mm |
| Adhesive | 3.0 | 3.0 | 0.10 mm |
| **Graphite spreader** | **1500** | **2** | variable |
| **Copper spreader** | **400** | **400** | variable |
| TIM | 3.0 | 3.0 | 0.10 mm |

### Boundary Conditions

| Surface | Condition |
|---------|-----------|
| z = 0, r < r_src | Neumann flux: q = Q / (pi * r_src^2) = 76.4 kW/m^2 |
| z = 0, r >= r_src | Adiabatic |
| z = H (glass back) | Convection h(T - T_inf) + radiation sigma*eps*(T^4 - T_inf^4) |
| r = 0 | Symmetry axis |
| r = R_gl | Adiabatic |

### Model Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Total power | Q | 6 W |
| Source radius | r_src | 5 mm |
| Spreader radius | R_sp | 37.5 mm |
| Glass radius | R_gl | 50 mm |
| Ambient temperature | T_amb | 25 °C |
| Convection coefficient | h | 10 W/m²K |
| Glass back emissivity | epsilon | 0.9 |

---

## Numerical Method

**Discretisation**  
Finite volume, cylindrical coordinates. Harmonic-mean interface conductivities at
material boundaries handle conductivity jumps without smearing. L'Hopital limit
enforced at the r = 0 symmetry axis.

**Nonlinear radiation BC**  
Linearised at each Newton iteration using h_rad = 4*sigma*eps*T^3 evaluated at the
previous temperature. Assembled directly into the sparse coefficient matrix.

**Linear system**  
SciPy `lil_matrix` → CSR format → `spsolve`. Approximately 90,000 DOF.
Converges in 4–6 Newton iterations to delta_T < 1e-4 °C.

**Grid**  
~1,500 radial nodes (refined at source edge r = 5 mm and spreader edge r = 37.5 mm),
~60 axial nodes (refined at each layer interface).

**Void region handling**  
Inner void nodes (r >= r_src, z <= t_TIM) are pinned to the spreader-base temperature
via a Dirichlet constraint T[j,i] = T[j_top,i], suppressing spurious conduction from
the spreader down into the air column. The node at exactly r = r_source is assigned
void (strict < in mat()) so the harmonic mean at the TIM/void interface is ~0.05 W/mK
rather than 3 W/mK.

---

## Key Results

### Validation vs FloEFD (corrected Sheet 1 data)

| Thickness | Material | FDM T_max [°C] | FloEFD T_max [°C] | Delta_T Error |
|-----------|----------|---------------|-------------------|---------------|
| 0.1 mm | Graphite | 97.80 | 98.06 | **0.4%** |
| 0.1 mm | Copper | 126.48 | 128.07 | **1.6%** |
| 0.4 mm | Graphite | 91.76 | 91.64 | **0.2%** |
| 0.4 mm | Copper | 96.26 | 95.80 | **0.7%** |
| 1.0 mm | Graphite | 92.02 | 92.05 | **0.04%** |
| 1.0 mm | Copper | 89.96 | 89.52 | **0.6%** |

Error normalised to delta_T above ambient (T_amb = 25°C).  
FloEFD run as half-symmetry model (180°, 3 W) — equivalent to full 360°/6 W.

### Material Crossover Summary

| Observation | Thickness range |
|-------------|-----------------|
| Graphite outperforms Copper | < ~0.75 mm |
| Curves cross over | ~0.75 mm |
| Copper outperforms Graphite | > ~0.75 mm |
| Diminishing returns (both materials) | > 0.4 mm (glass layer bottleneck) |

### Energy Balance (t = 0.1 mm Graphite)

| Path | Power | Share |
|------|-------|-------|
| Convection | ~3.24 W | 54% |
| Radiation | ~2.76 W | 46% |
| Balance error | — | < 0.5% |

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
```

---

*Python · SciPy · NumPy · Matplotlib · Cross-checked with Siemens FloEFD*

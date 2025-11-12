# bernabeu_test8.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path

# --- backend estável (evita erro do Qt) ---
os.environ.pop("QT_QPA_PLATFORM", None)
os.environ["MPLBACKEND"] = "TkAgg"

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- imports do módulo IHSetBernabeu (duas tentativas p/ compatibilidade) ---
try:
    from src.IHSetBernabeu.cal_bernabeu import cal_Bernabeu
except Exception:
    from IHSetBernabeu.cal_bernabeu import cal_Bernabeu  # pragma: no cover


# ============================== Paleta / Estilo ===============================
WATER       = "#a6cee3"   # azul água claro
SAND_LIGHT  = "#f4dcb8"   # areia clara (observado)
SAND_DARK   = "#d6b07a"   # areia escura (modelo final)
FULL_LINE   = "#da0000"   # vermelho (full)
HTL_LINE    = "#0025fa"   # azul (HTL)


# ============================== Helpers CSV ==================================
def read_xy_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Lê CSV com cabeçalho 'X,Y' (ou equivalentes) e devolve vetores finitos."""
    df = pd.read_csv(csv_path)
    cx = next(c for c in df.columns if c.lower() in ("x", "distance", "dist",
                                                     "cross_shore", "cross-shore"))
    cy = next(c for c in df.columns if c.lower() in ("y", "elevation", "z", "depth"))
    x = pd.to_numeric(df[cx], errors="coerce").to_numpy()
    y = pd.to_numeric(df[cy], errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


# ============================ Classe principal ===============================
class BernabeuTest:
    """
    Runner/plotter limpo para Bernabeu (2003), espelhando o fluxo do Dean v10.

    Auto-modo:
      - Se 'csv' for dado      -> calib_mode (slider X0 ativo)
      - Senão, se A/B/C/D dados -> A_mode
      - Caso contrário         -> D50_mode (pelo D50)

    Em calib_mode:
      - lê CSV, recalibra no intervalo [X0 → DoC] ao mover o slider e projeta HTL→DoC.
      - legenda única contendo A,B,C,D, R², RMSE; ordem de itens igual ao Dean.
    """

    def __init__(self, *, HTL, doc, D50, Hs50, Tp50, CM, hr,
                 A=None, B=None, C=None, D=None, csv=None):

        # ---- inputs principais ----
        self.HTL, self.doc = float(HTL), float(doc)
        self.D50, self.Hs50, self.Tp50 = D50, Hs50, Tp50
        self.CM, self.hr = float(CM), float(hr)
        self.A_user, self.B_user, self.C_user, self.D_user = A, B, C, D
        self.csv = csv

        # ---- modelo ----
        self.model = cal_Bernabeu(
            HTL=self.HTL, Hs50=self.Hs50, Tp50=self.Tp50,
            D50=self.D50, CM=self.CM, hr=self.hr, doc=self.doc
        )

        # ---- figure/axes ----
        self.fig, self.ax = plt.subplots(figsize=(12, 5.2))
        self.fig.subplots_adjust(bottom=0.22, right=0.82)
        self.ax.invert_yaxis()  # profundidade cresce para baixo

        # dados observados (opcionais)
        self.x_raw: np.ndarray | None = None
        self.y_raw: np.ndarray | None = None

        # handles que alimentam a legenda única
        self.htl_line = None
        self.obs_line = None
        self.doc_scatter = None
        self.calib_fill = None
        self.full_line = None

        self.slider = None

        self._auto_mode_and_run()
        plt.show()

    # ------------------------- seleção automática -------------------------
    def _auto_mode_and_run(self):
        if self.csv:
            self._calib_mode_setup()
            self._redraw_calib()
            self._make_slider()
        elif any(p is not None for p in (self.A_user, self.B_user, self.C_user, self.D_user)):
            self._theory_A_mode()
        else:
            self._theory_d50_mode()

    # -------------------------- modo teórico (D50) -------------------------
    def _theory_d50_mode(self):
        x_full, y_full, x_surf, y_surf, x_sh, y_sh = self._safe_run()
        # desenha e monta legenda (sem métricas; exibe D50 e A,B,C,D estimados)
        self._draw_all(x_full, y_full, x_surf, y_surf, x_sh, y_sh, show_obs=False)
        self._legend_metrics_bernabeu(
            A=self.model.A, B=self.model.B, C=self.model.C, D=self.model.D,
            r2=None, rmse=None, d50_mm=self.model.D50, show_calib_fill=False
        )

    # -------------------------- modo teórico (A/ABCD) ----------------------
    def _theory_A_mode(self):
        # aplica overrides do usuário
        if self.A_user is not None: self.model.change_A(self.A_user)
        if self.B_user is not None: self.model.change_B(self.B_user)
        if self.C_user is not None: self.model.change_C(self.C_user)
        if self.D_user is not None: self.model.change_D(self.D_user)

        x_full, y_full, x_surf, y_surf, x_sh, y_sh = self._safe_run()
        self._draw_all(x_full, y_full, x_surf, y_surf, x_sh, y_sh, show_obs=False)
        self._legend_metrics_bernabeu(
            A=self.model.A, B=self.model.B, C=self.model.C, D=self.model.D,
            r2=None, rmse=None, d50_mm=None, show_calib_fill=False
        )

    # ----------------------------- calibração CSV --------------------------
    def _calib_mode_setup(self):
        # lê CSV e popula o modelo
        self.model.add_data(self.csv)
        # espelhos locais só para facilitar limites/plots
        self.x_raw = self.model.x_raw
        self.y_raw = self.model.y_raw

    def _redraw_calib(self, new_x0: float | None = None):
        # 1) decide X0 (slider) e aplica no modelo
        if new_x0 is not None:
            self.model.x_drift = float(new_x0)
        X0_use = float(self.model.x_drift)

        # 2) calibra no segmento [X0 → Xdoc] ancorando em (X0, Y0_csv)
        x_full, y_full, (A,B,C,D,hr), X0, Y0, Xdoc = self.model.calibrate_segment_x(X0_use)

        # 3) limites de eixo
        x_min = float(np.nanmin(self.x_raw))   # primeiro ponto do perfil observado
        x_max = float(np.nanmax(self.x_raw))   # último ponto do perfil observado
        y_max = float(np.nanmin(self.y_raw))   # valor mais profundo (menor Y)
        y_min = float(np.nanmax(self.y_raw))   # valor mais raso (maior Y)
        
        # 4) limpa e redesenha base (água + HTL)
        self.ax.clear()
        self.ax.fill_between([x_min, x_max], self.model.HTL, y_min, color=WATER, zorder=1)
        (self.htl_line,) = self.ax.plot([x_min, x_max], [self.model.HTL, self.model.HTL],
                                        color=HTL_LINE, lw=1.6, zorder=2, label="High Tide level")

        # 5) observado do CSV (linha preta + areia clara)
        (self.obs_line,) = self.ax.plot(self.x_raw, self.y_raw, color="k", lw=1.6, zorder=3, label="Observed (CSV)")
        self.obs_fill = self.ax.fill_between(self.x_raw, self.y_raw, y_min, color=SAND_LIGHT, zorder=2)

        # 6) perfil Bernabeu final (vermelho) já calibrado + projetado HTL→DoC
        (self.full_line,) = self.ax.plot(x_full, y_full, "r--", lw=1.6, zorder=4)

        # 7) preencher SÓ o trecho calibrado [X0, Xdoc] com areia escura
        x_left, x_right = (X0, Xdoc) if X0 <= Xdoc else (Xdoc, X0)
        m_seg = (x_full >= x_left) & (x_full <= x_right)
        self.calib_fill = self.ax.fill_between(x_full[m_seg], y_full[m_seg], y_min,
                                            color=SAND_DARK, alpha=0.75, zorder=2.8)

        # 8) ponto DoC do CSV (marcador)
        i_doc = int(np.nanargmin(np.abs(self.y_raw - self.model.doc)))
        self.doc_scatter = self.ax.scatter([self.x_raw[i_doc]], [self.model.doc], c="k", s=18, zorder=5)

        # 9) eixos
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlabel("Cross-shore distance X [m]")
        self.ax.set_ylabel("Elevation / Depth [m]")

        # 10) métricas só no trecho calibrado
        rmse, r2 = self.model.metrics_segment(x_full, y_full, X0)
        self._rmse_last, self._r2_last = rmse, r2

        # 11) legenda única (ordem: HTL, Observed, DoC, Calibrated zone, Bernabeu + métricas)
        self._legend_metrics_bernabeu(
            A=A, B=B, C=C, D=D,
            r2=r2, rmse=rmse,
            d50_mm=None, show_calib_fill=True
        )

        self.fig.canvas.draw_idle()

    # ---------------------------- execução segura --------------------------
    def _safe_run(self):
        """Garante que run() sempre retorne 6 arrays (parciais vazios se não houver)."""
        out = self.model.run()
        if len(out) == 2:
            x_full, y_full = out
            x_surf = y_surf = x_shoal = y_shoal = np.array([])
        else:
            x_full, y_full, x_surf, y_surf, x_shoal, y_shoal = out
        return x_full, y_full, x_surf, y_surf, x_shoal, y_shoal

    # ------------------------------ desenho base ---------------------------
    def _draw_all(self, x_full, y_full, x_surf, y_surf, x_shoal, y_shoal, *, show_obs: bool):
        self.ax.clear()

        # ----- limites Y: topo = HTL - 2 m; fundo = DoC OU min(Y_obs) -----
        y_bottom = float(np.nanmax(self.y_raw)) if (show_obs and self.y_raw is not None) else float(self.doc)
        y_top = float(self.HTL - 2.0)

        # ----- limites X -----
        if show_obs and (self.x_raw is not None):
            x_min = float(np.nanmin(self.x_raw))
            x_max = float(np.nanmax(self.x_raw))
        else:
            # usa posição do DoC do perfil teórico e acrescenta +10 m
            idx_doc = int(np.nanargmin(np.abs(y_full - self.doc)))
            x_doc = float(x_full[idx_doc])
            x_min = float(np.nanmin(x_full))
            x_max = x_doc + 10.0

        # ----- água + HTL -----
        self.ax.fill_between([x_min, x_max], self.HTL, y_bottom, color=WATER, zorder=1)
        self.htl_line, = self.ax.plot([x_min, x_max], [self.HTL, self.HTL],
                                      color=HTL_LINE, lw=1.6, zorder=2, label="High Tide level")

        # ----- observado (CSV) -----
        self.obs_line = None
        if show_obs and (self.x_raw is not None):
            self.obs_line, = self.ax.plot(self.x_raw, self.y_raw, color="k", lw=1.6, zorder=4, label="Observed (CSV)")
            self.ax.fill_between(self.x_raw, self.y_raw, y_bottom, color=SAND_LIGHT, zorder=3)

        # ----- perfil completo de Bernabeu -----
        self.full_line, = self.ax.plot(x_full, y_full, ls="--", color=FULL_LINE, lw=1.8, zorder=6)
        self.calib_fill = self.ax.fill_between(x_full, y_full, y_bottom, color=SAND_DARK, alpha=0.75, zorder=5)

        # ----- marca DoC (x onde y≈doc) -----
        try:
            i_doc = int(np.nanargmin(np.abs(y_full - self.doc)))
            self.doc_scatter = self.ax.scatter([x_full[i_doc]], [self.doc], c="k", s=18, zorder=7)
        except Exception:
            self.doc_scatter = None

        # ----- eixos -----
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_top, y_bottom)  # topo->fundo
        self.ax.invert_yaxis()             # profundidade positiva para baixo
        self.ax.set_xlabel("Cross-shore distance X [m]")
        self.ax.set_ylabel("Elevation / Depth [m]")

        self.fig.canvas.draw_idle()

    # ------------------------------ legenda única --------------------------
    def _legend_metrics_bernabeu(self, *, A=None, B=None, C=None, D=None,
                                 r2=None, rmse=None, d50_mm=None, show_calib_fill=True):
        """
        Monta uma legenda única (ordem fixa):
        HTL, Observed, DoC, (D50), (Calibrated zone), Bernabeu Profile + métricas
        """
        lines, labels = [], []

        if self.htl_line is not None:
            lines.append(self.htl_line); labels.append("High Tide level")
        if self.obs_line is not None:
            lines.append(self.obs_line); labels.append("Observed (CSV)")
        if self.doc_scatter is not None:
            lines.append(self.doc_scatter); labels.append(f"DoC={self.doc:.1f} m")
        if d50_mm is not None:
            dummy_d50, = self.ax.plot([], [], alpha=0)
            lines.append(dummy_d50); labels.append(f"D50={float(d50_mm):.2f} mm")
        if show_calib_fill and self.calib_fill is not None:
            lines.append(self.calib_fill); labels.append("Calibrated zone")

        if self.full_line is not None:
            parts = ["Bernabeu Profile"]
            if A is not None:   parts.append(f"A={A:.3f}")
            if B is not None:   parts.append(f"B={B:.3f}")
            if C is not None:   parts.append(f"C={C:.3f}")
            if D is not None:   parts.append(f"D={D:.3f}")
            if r2 is not None:  parts.append(f"R²={r2:.3f}")
            if rmse is not None: parts.append(f"RMSE={rmse:.3f} m")
            self.full_line.set_label("\n".join(parts))
            lines.append(self.full_line); labels.append(self.full_line.get_label())

        leg = self.ax.legend(handles=lines, labels=labels,
                             loc="upper left", bbox_to_anchor=(1.02, 1),
                             frameon=True)
        leg.set_title("")
        try:
            leg._legend_box.align = "left"
        except Exception:
            pass

    # ------------------------------- slider --------------------------------
    def _make_slider(self):
        x_full, y_full, *_ = self.model.run()
        x_tr = self.model.x_at_transition(x_full, y_full)

        x0_min = float(np.nanmin(self.x_raw))
        x0_max = float(min(np.nanmax(self.x_raw), x_tr))

        ax_sl = self.fig.add_axes([0.10, 0.08, 0.78, 0.03])
        self.slider = Slider(ax_sl, "X0 (slider)", x0_min, x0_max,
                            valinit=float(self.model.x_drift), valstep=1.0)

        def _on_change(val):
            self._redraw_calib(new_x0=float(val))
        self.slider.on_changed(_on_change)

# ============================== Execução local ================================
if __name__ == "__main__":
    # Exemplo de uso — ajuste aqui os inputs que desejar
    HTL  = -2.0
    DoC  = 10.0
    D50  = 0.30   # mm
    Hs50 = 1.5    # m
    Tp50 = 8.0    # s
    CM   = 3.5    # m
    hr   = 2.0    # m

    # Se quiser teórico por A/ABCD em vez de D50, preencha aqui:
    A = None
    B = None
    C = None
    D = None

    # CSV opcional (ativa calib_mode automaticamente):
    CSV = None
    #CSV = str(Path(__file__).with_name("XY_PuertoChiquito_clean.csv"))

    BernabeuTest(
        HTL=HTL, doc=DoC, D50=D50, Hs50=Hs50, Tp50=Tp50,
        CM=CM, hr=hr, A=A, B=B, C=C, D=D, csv=CSV
    )

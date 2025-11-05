# bernabeu_tst_slider.py
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import matplotlib
os.environ.pop("QT_QPA_PLATFORM", None)
os.environ["MPLBACKEND"] = "TkAgg"
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon, Patch

# ---------- projeto ----------
root = Path(__file__).parent
sys.path.insert(0, str(root))
try:
    from cal_bernabeu import cal_Bernabeu
except Exception:
    from IHSetBernabeu.cal_bernabeu import cal_Bernabeu  # ajuste se usar pacote

WATER = "#a6cee3"; LIGHTSAND = "#f4dcb8"; DARKSAND = "#d6b07a"


class BernabeuTest:
    def __init__(self):
        # -------- entradas (ajuste aqui) --------
        self.HTL = -2.0
        self.doc = 8.0
        self.D50 = 0.30   # mm
        self.Hs50 = 1.0   # m
        self.Tp50 = 10.0  # s
        self.CM   = 3.0  # m (altura até o break)
        self.hr   = 1.5   # m
        self.csv  = str(root / "XY_PuertoChiquito_clean.csv")

        # modelo
        self.model = cal_Bernabeu(HTL=self.HTL, Hs50=self.Hs50, Tp50=self.Tp50,
                                  D50=self.D50, CM=self.CM, hr=self.hr,
                                  doc=self.doc, fit_mode="x")
        self.model.add_data(self.csv)

        self.xr = np.asarray(self.model.x_raw, float)
        self.yr = np.asarray(self.model.y_raw, float)

        self.xmin = float(np.nanmin(self.xr)); self.xmax = float(np.nanmax(self.xr))
        self.y_bottom = float(np.nanmax(self.yr))
        self.y_top    = float(min(np.nanmin(self.yr), self.HTL)) - 1.0

        # X(HTL) e X(DoC) do CSV
        self.Xhtl = self._x_at_y_csv(self.HTL)
        self.Xdoc = self._x_at_y_csv(self.doc)
        self.x0_init = np.clip(self.Xhtl, self.xmin, self.Xdoc - 1e-6)

        # -------- figura/layout --------
        self.fig, self.ax = plt.subplots(figsize=(12, 6.8))
        plt.subplots_adjust(right=0.80, bottom=0.16)

        # água
        self.ax.fill_between([self.xmin, self.xmax], self.HTL, self.y_bottom,
                             color=WATER, alpha=0.7, zorder=0)
        (self.htl_line,) = self.ax.plot([self.xmin, self.xmax], [self.HTL, self.HTL],
                                        color="blue", lw=1.8, label="High Tide level", zorder=1)

        # observado
        order = np.argsort(self.xr)
        self.ax.fill_between(self.xr[order], self.yr[order], self.y_bottom,
                             color=LIGHTSAND, alpha=0.95, zorder=2)
        (self.obs_line,) = self.ax.plot(self.xr, self.yr, color="black", lw=1.8,
                                        label="Observed (CSV)", zorder=4)

        # Bernabeu (curva e polígonos)
        (self.b_line,) = self.ax.plot([], [], "r--", lw=1.8,
                                      label="Bernabeu Profile", zorder=5)
        self.poly_cal = None  # área escura somente no trecho calibrado

        # DoC
        self.doc_pt = self.ax.scatter([self.Xdoc], [self.doc], s=38, c="k", zorder=6)

        # limites e eixos
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.y_top, self.y_bottom)
        self.ax.invert_yaxis()
        self.ax.set_xlabel("Cross-shore distance X [m]")
        self.ax.set_ylabel("Elevation / Depth [m]")
        self.ax.grid(True, linestyle=":", lw=0.5)

        # legenda (ordem e textos)
        calib_proxy = Patch(facecolor=DARKSAND, edgecolor="none", alpha=0.95, label="Calibrated zone")
        d50_proxy   = Patch(facecolor="none", edgecolor="none", label=f"D50={self.D50:.2f} mm")
        self._legend_labels = ["High Tide level", "Observed (CSV)",
                               f"DoC={self.doc:.1f} m", f"D50={self.D50:.2f} mm",
                               "Calibrated zone", "Bernabeu Profile"]
        handles = [self.htl_line, self.obs_line, self.doc_pt, d50_proxy, calib_proxy, self.b_line]
        leg = self.ax.legend(handles=handles, labels=self._legend_labels,
                             loc="upper left", bbox_to_anchor=(1.02, 1.00),
                             frameon=False)
        # referência ao texto do item “Bernabeu Profile”
        idx = self._legend_labels.index("Bernabeu Profile")
        self.b_label_text = leg.get_texts()[idx]

        # slider (debounce)
        ax_sl = self.fig.add_axes([0.15, 0.04, 0.70, 0.035])
        self.slider = Slider(ax=ax_sl, label="X0 Drift",
                             valmin=self.Xhtl, valmax=self.Xdoc - 1e-6,
                             valinit=self.x0_init, valstep=0.5)
        self._pending_x0 = None
        self._debounce = self.fig.canvas.new_timer(interval=40)
        try: self._debounce.single_shot = True
        except Exception: pass
        self._debounce.add_callback(self._fire_update)
        self.slider.on_changed(self._on_slider)

        # primeira atualização
        self._really_update(self.x0_init)

    # ---------- utilitários ----------
    def _x_at_y_csv(self, yq: float) -> float:
        xr, yr = self.xr, self.yr
        d = yr - yq; s = np.sign(d)
        cross = np.where(s[:-1] * s[1:] <= 0)[0]
        if cross.size:
            i = cross[0]
            x0, x1 = xr[i], xr[i+1]; y0, y1 = yr[i], yr[i+1]
            if y1 == y0: return float(x0)
            return float(x0 + (yq - y0) * (x1 - x0) / (y1 - y0))
        k = int(np.argmin(np.abs(d)))
        if k == 0:
            x0, x1, y0, y1 = xr[0], xr[1], yr[0], yr[1]
        else:
            x0, x1, y0, y1 = xr[-2], xr[-1], yr[-2], yr[-1]
        if y1 == y0: return float(x0)
        return float(x0 + (yq - y0) * (x1 - x0) / (y1 - y0))

    @staticmethod
    def _build_fill_verts(x, y, y_bottom):
        return np.vstack([np.column_stack([x, y]),
                          [x[-1], y_bottom],
                          [x[0],  y_bottom]])

    def _rmse_on_segment(self, xs, ys):
        if len(xs) < 5: return np.nan
        m = (self.xr >= xs.min()) & (self.xr <= xs.max())
        if not np.any(m): return np.nan
        y_csv = np.interp(xs, self.xr[m], self.yr[m])
        return float(np.sqrt(np.mean((y_csv - ys) ** 2)))

    def _r2_on_segment(self, xs, ys):
        if len(xs) < 5: return np.nan
        m = (self.xr >= xs.min()) & (self.xr <= xs.max())
        if not np.any(m): return np.nan
        y_csv = np.interp(xs, self.xr[m], self.yr[m])
        ss_res = float(np.sum((y_csv - ys) ** 2))
        ss_tot = float(np.sum((y_csv - np.mean(y_csv)) ** 2))
        if ss_tot <= 0: return np.nan
        return 1.0 - ss_res / ss_tot

    # ---------- slider debounce ----------
    def _on_slider(self, val):
        self._pending_x0 = float(val)
        try: self._debounce.stop()
        except Exception: pass
        self._debounce.start()

    def _fire_update(self):
        if self._pending_x0 is None:
            try: self._debounce.stop()
            except Exception: pass
            return
        x0 = self._pending_x0
        self._pending_x0 = None
        self._really_update(x0)
        self.fig.canvas.draw_idle()
        try: self._debounce.stop()
        except Exception: pass

    # ---------- atualização principal ----------
    def _really_update(self, x0_abs):
        # calibre somente na janela [Y0..DoC], prolongue visualmente até HTL
        try:
            x_seg, y_seg, pars, X0, Y0, Xdoc = self.model.calibrate_segment_x(x0_abs)
        except Exception:
            self.b_line.set_data([], [])
            if self.poly_cal is not None:
                tiny = 1e-6
                self.poly_cal.set_xy(np.array([[0,0],[tiny,0],[0,tiny]]))
            return

        # prolongamento gráfico até HTL (sem fill)
        if Y0 > self.HTL:
            n_up = max(20, int(200 * (Y0 - self.HTL) / max(1e-6, (self.doc - self.HTL))))
            y_up = np.linspace(self.HTL, Y0, n_up, endpoint=False)
            # a curva (x(h),y(h)) está em self.model.run(); para extrapolar suavemente,
            # aproxime por uma tangente linear próxima de Y0 usando dois pontos iniciais do segmento:
            x2p, y2p = x_seg[:2], y_seg[:2]
            if len(x2p) == 2 and (y2p[1] - y2p[0]) != 0:
                m = (x2p[1] - x2p[0]) / (y2p[1] - y2p[0])
                x_up = X0 + m * (y_up - Y0)
            else:
                x_up = np.full_like(y_up, X0)
            x_plot = np.concatenate([x_up, x_seg])
            y_plot = np.concatenate([y_up, y_seg])
        else:
            x_plot, y_plot = x_seg, y_seg

        self.b_line.set_data(x_plot, y_plot)

        # fill somente na zona calibrada (X0..Xdoc)
        order = np.argsort(x_seg)
        xs, ys = x_seg[order], y_seg[order]
        verts = self._build_fill_verts(xs, ys, self.y_bottom)
        if self.poly_cal is None:
            self.poly_cal = Polygon(verts, closed=True,
                                    facecolor=DARKSAND, edgecolor="none",
                                    alpha=0.95, zorder=3)
            self.ax.add_patch(self.poly_cal)
        else:
            self.poly_cal.set_xy(verts)

        # DoC (ponto)
        self.doc_pt.set_offsets(np.c_[[Xdoc], [self.doc]])

        # métricas no item “Bernabeu Profile”
        rmse = self._rmse_on_segment(xs, ys)
        r2   = self._r2_on_segment(xs, ys)
        A, B, C, D, hr = pars
        lbl = ["Bernabeu Profile",
               f"A={A:.4f}", f"B={B:.4f}",
               f"C={C:.4f}", f"D={D:.4f}", f"hr={hr:.3f}"]
        if np.isfinite(r2):   lbl.append(f"R²={r2:.3f}")
        if np.isfinite(rmse): lbl.append(f"RMSE={rmse:.3f} m")
        self.b_label_text.set_text("\n".join(lbl))


if __name__ == "__main__":
    BernabeuTest()
    plt.show()

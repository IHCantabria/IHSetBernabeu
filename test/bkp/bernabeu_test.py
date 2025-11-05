import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', 'src')))
from IHSetBernabeu.cal_bernabeu import cal_Bernabeu

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
import numpy as np

class BernabeuTest:

    def __init__(self):
        self.HTL  = -2.0   # High tide level (vertical shift)
        self.Hs50 =  2.0   # Significant wave height
        self.Tp50 =  8.0   # Peak wave period
        self.D50  =  0.3   # Median grain size [mm]
        self.CM   =  1.5   # Depth at start of shoaling
        self.hr   =  0.5   # Inflection depth between surf and shoaling
        self.doc  = 6.0   # Depth of closure
        self.csv  = "XY_PuertoChiquito_clean.csv"

        # NEW: choose calibration mode: "x" (default) or "y"
        self.fit_mode = "x"

        # 1. Instantiate model
        self.model = cal_Bernabeu(self.HTL, self.Hs50, self.Tp50, self.D50,
                                  self.CM, self.hr, self.doc, fit_mode=self.fit_mode)

    def plot(self):
        # 2 - Theoretical (no CSV)
        x_theo, y_theo = self.model.from_D50(self.D50)

        # Colors
        LIGHTSAND = "#f4dcb8"
        DARKSAND  = "#d6b07a"
        WATER     = "#a6cee3"

        # 8 - Calibrated against CSV
        df_field = pd.read_csv(self.csv, dtype={'X': float, 'Y': float})
        x_obs = pd.to_numeric(df_field['X'], errors='coerce').to_numpy()
        y_obs = pd.to_numeric(df_field['Y'], errors='coerce').to_numpy()
        # keep sign convention as-is; only ensure finite
        mfin = np.isfinite(x_obs) & np.isfinite(y_obs)
        x_obs = x_obs[mfin]; y_obs = y_obs[mfin]

        (x_cal, y_cal) = self.model.add_data(self.csv)

        # Plot
        fig, ax = plt.subplots(figsize=(10,6))
        # background water
        all_x = np.concatenate([x_obs, x_cal])
        all_y = np.concatenate([y_obs, y_cal])
        x_min, x_max = float(np.nanmin(all_x)), float(np.nanmax(all_x))
        y_bottom = float(np.nanmax(all_y))
        y_top = float(min(np.nanmin(all_y), self.HTL)) - 1.0

        ax.fill_between([x_min, x_max], self.model.HTL, y_bottom, color=WATER, alpha=0.7, zorder=1)
        # CSV fill + line
        ax.fill_between(x_obs, y_obs, y_bottom, color=LIGHTSAND, zorder=2)
        line_csv, = ax.plot(x_obs, y_obs, '-', color='black', lw=1.6, label='Observed (CSV)', zorder=4)
        # Calibrated fill + line
        fill_cal = ax.fill_between(x_cal, y_cal, y_bottom, color=DARKSAND, alpha=0.7, zorder=3)
        line_cal, = ax.plot(x_cal, y_cal, '--', color='red', lw=1.8, label='Bernabeu Profile', zorder=5)
        # HTL
        line_HTL, = ax.plot([x_min, x_max], [self.model.HTL, self.model.HTL], color="blue", lw=1.5, label="High Tide level", zorder=0)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_top, y_bottom)
        ax.set_xlabel("Cross-shore distance X [m]")
        ax.set_ylabel("Elevation / Depth Y [m]")
        ax.invert_yaxis()
        ax.grid(True, linestyle=':', lw=0.5)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        ax.set_title(f"Bernabeu Profile (fit_mode='{self.fit_mode}')")

        # ---------------- slider x0 (drift) with recalibration ----------------
        # estimate x0 range from CSV
        x0_min, x0_max = float(np.nanmin(x_obs)), float(np.nanmax(x_obs))
        init_x0 = float(getattr(self.model, "x_drift", np.nanmin(x_obs)))

        ax_sl = plt.axes([0.09, 0.03, 0.72, 0.03])
        slider = Slider(ax=ax_sl, label="x0 (m) – drift (recalibration)", valmin=x0_min, valmax=x0_max,
                        valinit=init_x0, valstep=0.5)

        def on_change(val):
            x0 = slider.val
            try:
                x_new, y_new = self.model.shift_HTL(x0, recalibrate=True)
            except Exception as e:
                print(f"[Warning] Shift failed x0={x0:.2f}: {e}")
                return

            # update display
            line_cal.set_data(x_new, y_new)
            nonlocal fill_cal
            fill_cal.remove()
            fill_cal = ax.fill_between(x_new, y_new, y_bottom, color=DARKSAND, alpha=0.7, zorder=3)
            fig.canvas.draw_idle()

        slider.on_changed(on_change)

        plt.tight_layout()
        plt.show()

# Run if executed as script
if __name__ == "__main__":
    BernabeuTest().plot()

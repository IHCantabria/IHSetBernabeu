# cal_bernabeu.py
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .bernabeu import Bernabeu
from IHSetUtils import wMOORE


class cal_Bernabeu(object):
    """
    Calibrador/runner para o perfil de Bernabeu (2003).
    Agora com:
      - fit_mode: "x" (padrão) ou "y"
      - calibrate_segment_x(x0_abs): calibração usando janela [Y0..DoC]
        definida por um X0 absoluto (slider), e retorno do segmento calibrado.
    """

    import warnings

    def __init__(self, HTL, Hs50, Tp50, D50, CM, hr, doc, fit_mode: str = "x", strict=False):
        # --- checks rápidos que permanecem iguais ---
        if not (-17.0 <= HTL <= 17.0):
            raise ValueError("HTL fora de [-17, 17] m")
        if not (4.0 <= Tp50 <= 20.0):
            raise ValueError("Tp50 fora de [4, 20] s")
        if not (0.06 <= D50 <= 4.0):
            raise ValueError("D50 fora de [0.06, 4.0] mm")
        if not (HTL < doc <= HTL + 20.0):
            raise ValueError("DoC deve estar abaixo de HTL e até ~HTL+20 m")

        # --- Hs50: intervalo relaxado ---
        hs_lo, hs_hi = 0.05, 8.0
        if strict:
            if not (hs_lo <= Hs50 <= hs_hi):
                raise ValueError(f"Hs50 fora de [{hs_lo}, {hs_hi}] m")
        else:
            if Hs50 < hs_lo or Hs50 > hs_hi:
                warnings.warn(f"Hs50={Hs50:.2f} fora de [{hs_lo}, {hs_hi}] m; ajustando.")
                Hs50 = min(max(Hs50, hs_lo), hs_hi)

        # --- hr: intervalo relaxado dependente de HTL/CM/DoC ---
        lower_hr = max(HTL + 0.05, 0.10)
        upper_hr = 0.90 * CM
        if doc is not None:
            upper_hr = min(upper_hr, doc - 0.10)

        if strict:
            if not (lower_hr < hr < upper_hr):
                raise ValueError(f"hr fora de ({lower_hr:.2f}, {upper_hr:.2f}) m")
        else:
            if not (lower_hr < hr < upper_hr):
                warnings.warn(f"hr={hr:.2f} fora de ({lower_hr:.2f}, {upper_hr:.2f}) m; ajustando.")
                hr = min(max(hr, lower_hr + 1e-6), upper_hr - 1e-6)

        # (opcional) manter controle de arrebentação, mas um pouco mais amplo:
        gamma_break = CM / Hs50
        if strict:
            if not (0.5 <= gamma_break <= 2.0):
                raise ValueError("CM/Hs50 fora de 0.5–2.0")
        else:
            if not (0.3 <= gamma_break <= 3.0):
                warnings.warn(f"CM/Hs50={gamma_break:.2f} fora de 0.3–3.0 (continuando).")

        # ... seguir com a inicialização normal
        self.HTL, self.Hs50, self.Tp50, self.D50 = HTL, Hs50, Tp50, D50
        self.CM, self.hr, self.doc = CM, hr, doc
        self.fit_mode = fit_mode

        # --------- parâmetros base ---------
        self.HTL, self.Hs50, self.Tp50 = float(HTL), float(Hs50), float(Tp50)
        self.D50, self.CM, self.hr = float(D50), float(CM), float(hr)
        self.doc = float(doc)

        self.fit_mode = (fit_mode or "x").lower().strip()
        if self.fit_mode not in ("x", "y"):
            raise ValueError("fit_mode deve ser 'x' ou 'y'.")

        # dados/estados
        self.x_raw = self.y_raw = None
        self.x_obs = self.y_obs = self.y_obs_rel = None
        self.data = False
        self.x_drift = 0.0

        # resultados correntes (perfil completo)
        self.x_full = self.y_full = None
        self.x1 = self.y1 = self.x2 = self.y2 = None

        self.params()
        self.def_hvec()

    # ---------- parâmetros de Bernabeu a partir de D50, Hs50, Tp50 ----------
    def params(self):
        ws = wMOORE(self.D50 / 1000.0)
        gamma = self.Hs50 / (ws * self.Tp50)
        A_raw = 0.21 - 0.02 * gamma
        self.A = max(A_raw, 1e-3)
        self.B = 0.89 * np.exp(-1.24 * gamma)
        self.C = 0.06 + 0.04 * gamma
        self.D = 0.22 * np.exp(-0.83 * gamma)

    def def_hvec(self):
        # vetor vertical h (positivo para baixo) cobrindo Shoaling+Surf e DoC
        self.h = np.arange(0.0, self.CM + (self.doc - self.HTL) + 1e-9, 0.001)

    def def_xo(self, A=None, B=None, C=None, D=None, hr=None):
        A = self.A if A is None else A
        B = self.B if B is None else B
        C = self.C if C is None else C
        D = self.D if D is None else D
        hr = self.hr if hr is None else hr
        return ((hr + self.CM) / A) ** 1.5 - (hr / C) ** 1.5 + (B / (A ** 1.5)) * (hr + self.CM) ** 3 - (D / (C ** 1.5)) * hr ** 3

    # ---------- execução do perfil ----------
    def run(self, A=None, B=None, C=None, D=None, hr=None, x_drift=None):
        A = self.A if A is None else A
        B = self.B if B is None else B
        C = self.C if C is None else C
        D = self.D if D is None else D
        hr = self.hr if hr is None else hr

        xo = self.def_xo(A, B, C, D, hr)
        x_raw, x1_raw, x2_raw, h2 = Bernabeu(A, B, C, D, self.CM, self.h, xo)

        xd = self.x_drift if x_drift is None else float(x_drift)
        self.x_full = x_raw + xd
        self.y_full = self.h + self.HTL

        mask_surf = self.h <= (hr + self.CM)

        # Surf (h <= hr+CM)
        self.x1 = self.x_full[mask_surf]
        self.y1 = self.y_full[mask_surf]

        # Shoaling (h > hr+CM). h2 já é self.h[~mask_surf] - CM
        self.x2 = self.x_full[~mask_surf]
        self.y2 = h2 + self.CM + self.HTL  # ← sem aplicar máscara novamente

        return self.x_full, self.y_full

    # ---------- leitura de dados ----------
    def add_data(self, path_csv):
        df = pd.read_csv(path_csv, dtype={"X": float, "Y": float})
        xr = pd.to_numeric(df["X"], errors="coerce").to_numpy()
        yr = pd.to_numeric(df["Y"], errors="coerce").to_numpy()
        m = np.isfinite(xr) & np.isfinite(yr)
        self.x_raw, self.y_raw = xr[m], yr[m]

        # shoreline (x_drift = X @ Y==HTL) — seguro a partir de Y
        idx = np.argsort(self.y_raw)
        self.x_drift = float(np.interp(self.HTL, self.y_raw[idx], self.x_raw[idx]))

        # janela vertical [HTL..DoC]
        mask = (self.y_raw >= self.HTL) & (self.y_raw <= self.doc)
        x_cut, y_cut = self.x_raw[mask], self.y_raw[mask]
        self.x_obs = x_cut - self.x_drift
        self.y_obs = y_cut
        self.y_obs_rel = self.y_obs - self.HTL
        self.data = True

        self.calibrate()  # calibração inicial com janela completa
        return self.x_full, self.y_full

    # ---------- helpers ----------
    def y_at_x_csv(self, xq_abs: float) -> float:
        """Interpolação robusta de Y(X) no CSV original (absoluto)."""
        xr, yr = self.x_raw, self.y_raw
        if xr is None: raise ValueError("CSV ainda não carregado.")
        # ordenar por X
        idx = np.argsort(xr)
        xr, yr = xr[idx], yr[idx]
        return float(np.interp(xq_abs, xr, yr))

    def _rebuild_obs_keep_drift(self):
        """Reconstrói x_obs/y_obs dentro de [HTL..DoC] mantendo self.x_drift atual."""
        xr, yr = self.x_raw, self.y_raw
        m = (yr >= self.HTL) & (yr <= self.doc) & np.isfinite(xr) & np.isfinite(yr)
        x_cut, y_cut = xr[m], yr[m]
        self.x_obs = x_cut - float(self.x_drift)
        self.y_obs = y_cut
        self.y_obs_rel = self.y_obs - self.HTL

    # ---------- calibração padrão (janela completa [HTL..DoC]) ----------
    def calibrate(self):
        if not self.data:
            raise ValueError("Use add_data() antes de calibrar.")
        return self._least_squares_fit()

    # ---------- calibração por X0 (janela [Y0..DoC]) ----------
    def calibrate_segment_x(self, x0_abs: float):
        """
        Fixa X0 absoluto (slider), encontra Y0=Ycsv(X0),
        usa somente a janela [Y0..DoC] para a calibração.
        Retorna:
          x_seg, y_seg, (A,B,C,D,hr), X0, Y0, Xdoc
        """
        if not self.data:
            raise ValueError("Use add_data() antes de calibrar.")

        # fixa drift (origem em X0)
        self.x_drift = float(x0_abs)
        self._rebuild_obs_keep_drift()

        # Y0 do CSV no X0
        Y0 = self.y_at_x_csv(x0_abs)
        if not (self.HTL - 1e-6 <= Y0 <= self.doc + 1e-6):
            # fora da janela vertical: ainda assim tente encaixar limitando
            Y0 = np.clip(Y0, self.HTL, self.doc)

        # cria máscaras de janela para observados: h>=h0
        h0 = float(Y0 - self.HTL)
        mwin = self.y_obs_rel >= h0
        if np.count_nonzero(mwin) < 5:
            # pouco dado — volta para janela total (robustez)
            mwin = np.ones_like(self.y_obs_rel, dtype=bool)

        # guarda cópias das observações “janeladas”
        x_obs_all, y_obs_rel_all, y_obs_all = self.x_obs, self.y_obs_rel, self.y_obs
        self.x_obs = x_obs_all[mwin]
        self.y_obs_rel = y_obs_rel_all[mwin]
        self.y_obs = y_obs_all[mwin]

        # ajusta parâmetros
        self._least_squares_fit()

        # reconstrói perfil e recorta o segmento calibrado [Y0..DoC]
        xf, yf = self.run()
        mseg = (yf >= Y0 - 1e-9) & (yf <= self.doc + 1e-9)
        x_seg, y_seg = xf[mseg], yf[mseg]

        # devolve observações completas
        self.x_obs, self.y_obs_rel, self.y_obs = x_obs_all, y_obs_rel_all, y_obs_all

        # X(DoC) do CSV
        # ordenar por Y para inversão
        idx = np.argsort(self.y_raw)
        Xdoc = float(np.interp(self.doc, self.y_raw[idx], self.x_raw[idx]))
        return x_seg, y_seg, (self.A, self.B, self.C, self.D, self.hr), float(x0_abs), float(Y0), float(Xdoc)

    # ---------- LSQ comum para fit_mode 'x' e 'y' ----------
    def _least_squares_fit(self):
        def model_x_of_h(A, B, C, D, hr):
            xo = self.def_xo(A, B, C, D, hr)
            mask1 = self.h <= (hr + self.CM)
            h1 = self.h[mask1]
            x1 = (h1 / A) ** 1.5 + B / (A ** 1.5) * h1 ** 3
            h2 = self.h[~mask1] - self.CM
            x2 = (h2 / C) ** 1.5 + D / (C ** 1.5) * h2 ** 3 + xo
            x_model = np.empty_like(self.h)
            x_model[mask1] = x1
            x_model[~mask1] = x2
            return x_model

        def residuals_x(plog):
            A, B, C, D = np.exp(plog[:4]); hr = plog[4]
            x_model = model_x_of_h(A, B, C, D, hr)
            m = np.isfinite(self.x_obs) & np.isfinite(self.y_obs_rel)
            if not np.any(m): return np.array([1e6])
            yr = self.y_obs_rel[m]; xr = self.x_obs[m]
            idx = np.argsort(yr)
            yr, xr = yr[idx], xr[idx]
            mm = yr >= 0.0
            if not np.any(mm): return np.array([1e6])
            yr, xr = yr[mm], xr[mm]
            x_obs_interp = np.interp(self.h, yr, xr, left=np.nan, right=np.nan)
            res = x_model - x_obs_interp
            return res[np.isfinite(res)]

        def residuals_y(plog):
            A, B, C, D = np.exp(plog[:4]); hr = plog[4]
            x_model = model_x_of_h(A, B, C, D, hr)
            y_model = self.h + self.HTL
            m = np.isfinite(self.x_obs) & np.isfinite(self.y_obs)
            if not np.any(m): return np.array([1e6])
            xo, yo = self.x_obs[m], self.y_obs[m]
            idxm = np.argsort(x_model)
            xm, ym = x_model[idxm], y_model[idxm]
            xmin, xmax = np.nanmin(xm), np.nanmax(xm)
            mm = (xo >= xmin) & (xo <= xmax)
            if not np.any(mm): return np.array([1e6])
            y_pred = np.interp(xo[mm], xm, ym)
            return y_pred - yo[mm]

        plog0 = np.array([
            np.log(max(self.A, 1e-4)),
            np.log(max(self.B, 1e-6)),
            np.log(max(self.C, 1e-4)),
            np.log(max(self.D, 1e-6)),
            float(self.hr)
        ])
        lb = [np.log(1e-5), np.log(1e-8), np.log(1e-5), np.log(1e-8), 0.0]
        ub = [np.log(1e+1), np.log(1e+2), np.log(1e+1), np.log(1e+2), float(self.h.max())]
        fun = residuals_x if self.fit_mode == "x" else residuals_y

        res = least_squares(fun, plog0, bounds=(lb, ub),
                            loss="huber", f_scale=0.1,
                            xtol=1e-8, ftol=1e-8, max_nfev=4000)
        self.A, self.B, self.C, self.D = np.exp(res.x[:4])
        self.hr = float(res.x[4])
        self.def_hvec()
        return self.run()

# cal_bernabeu.py — versão compatível com A,B,C,D opcionais e setters completos
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# No seu repositório o nome é bernabeu_mod; ajuste aqui se for "Bernabeu"
from .bernabeu import bernabeu_mod
from IHSetUtils import wMOORE


class cal_Bernabeu:
    """
    Runner/Calibrador do perfil de Bernabeu (2003).

    - Aceita coeficientes A,B,C,D opcionalmente no __init__. Se ausentes,
      são computados a partir de D50, Hs50, Tp50 (como no paper).
    - Métodos de variação: from_D50, from_Hs50, from_Tp50, change_HTL,
      change_doc, change_CM, change_hr, change_A/B/C/D.
    - 'fit_mode' permite calibração em 'x' (padrão) ou 'y' quando CSV for usado.
    - 'calibrate_segment_x(x0_abs)' calibra na janela [Y0..DoC] fixando X0 absoluto.
    """

    def __init__(
        self,
        HTL: float,
        Hs50: float,
        Tp50: float,
        D50: float,
        CM: float,
        hr: float,
        doc: float,
        *,
        A: float | None = None,
        B: float | None = None,
        C: float | None = None,
        D: float | None = None,
        fit_mode: str = "x",
        strict: bool = False,
    ):
        # ---------- validações rápidas (intervalos levemente relaxados) ----------
        if not (-17.0 <= HTL <= 17.0):
            raise ValueError("HTL fora de [-17, 17] m")
        if not (4.0 <= Tp50 <= 20.0):
            raise ValueError("Tp50 fora de [4, 20] s")
        if not (0.06 <= D50 <= 4.0):
            raise ValueError("D50 fora de [0.06, 4.0] mm")
        if not (HTL < doc <= HTL + 20.0):
            raise ValueError("DoC deve estar abaixo de HTL e até ~HTL+20 m")

        # Hs50
        hs_lo, hs_hi = 0.05, 8.0
        if strict:
            if not (hs_lo <= Hs50 <= hs_hi):
                raise ValueError(f"Hs50 fora de [{hs_lo}, {hs_hi}] m")
        else:
            if Hs50 < hs_lo or Hs50 > hs_hi:
                warnings.warn(f"Hs50={Hs50:.2f} fora de [{hs_lo}, {hs_hi}] m; ajustando.")
                Hs50 = min(max(Hs50, hs_lo), hs_hi)

        # hr dependente de CM/HTL/DoC
        lower_hr = max(HTL + 0.05, 0.10)
        upper_hr = min(0.90 * CM, doc - 0.10)
        if strict:
            if not (lower_hr < hr < upper_hr):
                raise ValueError(f"hr fora de ({lower_hr:.2f}, {upper_hr:.2f}) m")
        else:
            if not (lower_hr < hr < upper_hr):
                warnings.warn(f"hr={hr:.2f} fora de ({lower_hr:.2f}, {upper_hr:.2f}) m; ajustando.")
                hr = min(max(hr, lower_hr + 1e-6), upper_hr - 1e-6)

        gamma_break = CM / Hs50
        if strict:
            if not (0.5 <= gamma_break <= 2.0):
                raise ValueError("CM/Hs50 fora de 0.5–2.0")
        else:
            if not (0.3 <= gamma_break <= 3.0):
                warnings.warn(f"CM/Hs50={gamma_break:.2f} fora de 0.3–3.0 (continuando).")

        # ---------- estados ----------
        self.HTL, self.Hs50, self.Tp50 = float(HTL), float(Hs50), float(Tp50)
        self.D50, self.CM, self.hr = float(D50), float(CM), float(hr)
        self.doc = float(doc)
        self.fit_mode = (fit_mode or "x").lower().strip()
        if self.fit_mode not in ("x", "y"):
            raise ValueError("fit_mode deve ser 'x' ou 'y'.")

        # dados observados (quando houver CSV)
        self.x_raw = self.y_raw = None
        self.x_obs = self.y_obs = self.y_obs_rel = None
        self.x_drift = 0.0
        self.data = False

        # resultados correntes (perfil completo e segmentos)
        self.x_full = self.y_full = None
        self.x1 = self.y1 = self.x2 = self.y2 = None

        # parâmetros (ou fornecidos, ou calculados)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        if any(p is None for p in (self.A, self.B, self.C, self.D)):
            self._params_from_grain()  # calcula a,b,c,d
        # vetor vertical de h
        self._def_hvec()

    # ---------- parâmetros a partir de D50/Hs50/Tp50 ----------
    def _params_from_grain(self):
        ws = wMOORE(self.D50 / 1000.0)         # m/s
        gamma = self.Hs50 / (ws * self.Tp50)   # adimensional
        A_raw = 0.21 - 0.02 * gamma
        self.A = self.A if self.A is not None else max(A_raw, 1e-3)
        self.B = self.B if self.B is not None else 0.89 * np.exp(-1.24 * gamma)
        self.C = self.C if self.C is not None else 0.06 + 0.04 * gamma
        self.D = self.D if self.D is not None else 0.22 * np.exp(-0.83 * gamma)

    def _def_hvec(self):
        # h: profundidade positiva para baixo, cobrindo surf+shoaling até DoC (relativo ao HTL)
        span = (self.doc - self.HTL)  # diferença vertical HTL→DoC
        self.h = np.arange(0.0, self.CM + span + 1e-9, 0.001)

    def _def_xo(self, A=None, B=None, C=None, D=None, hr=None):
        A = self.A if A is None else A
        B = self.B if B is None else B
        C = self.C if C is None else C
        D = self.D if D is None else D
        hr = self.hr if hr is None else hr
        return ((hr + self.CM) / A) ** 1.5 - (hr / C) ** 1.5 + (B / (A ** 1.5)) * (hr + self.CM) ** 3 - (D / (C ** 1.5)) * hr ** 3

    # ---------- execução do perfil teórico ----------
    def run(self, A=None, B=None, C=None, D=None, hr=None, x_drift=None):
        # aplica overrides opcionais
        if A is not None:  self.A  = float(A)
        if B is not None:  self.B  = float(B)
        if C is not None:  self.C  = float(C)
        if D is not None:  self.D  = float(D)
        if hr is not None: self.hr = float(hr)
        if x_drift is not None: self.x_drift = float(x_drift)

        xo = self._def_xo()

        # Saída “bruta” do seu solver de Bernabeu (perfil em profundidades h):
        # x_raw: perfil completo com mesmo tamanho de self.h
        # (x1_raw/x2_raw podem vir com outros tamanhos; evitamos usa-los)
        x_raw, _x1_raw, _x2_raw, _h2 = bernabeu_mod(self.A, self.B, self.C, self.D, self.CM, self.h, xo)

        # Constrói perfil completo em coordenadas do gráfico:
        x_full = x_raw + self.x_drift            # deslocamento horizontal
        y_full = self.h + self.HTL               # y absoluto (eixo será invertido no plot)

        # Partições a partir do perfil completo (garante alinhamento)
        x_surf,  y_surf, x_shoal, y_shoal = self._split_parts_from_full(x_full, y_full)

        # Sempre retornar 6 vetores
        return x_full, y_full, x_surf, y_surf, x_shoal, y_shoal

    # ---------- Setters/variadores (recomputam parâmetros, se necessário) ----------
    def from_D50(self, D50):
        self.D50 = float(D50)
        self._params_from_grain()
        return self.run()

    def from_Hs50(self, Hs50):
        self.Hs50 = float(Hs50)
        self._params_from_grain()
        return self.run()

    def from_Tp50(self, Tp50):
        self.Tp50 = float(Tp50)
        self._params_from_grain()
        return self.run()

    def change_HTL(self, HTL):
        self.HTL = float(HTL)
        self._def_hvec()
        return self.run()

    def change_doc(self, doc):
        self.doc = float(doc)
        self._def_hvec()
        return self.run()

    def change_CM(self, CM):
        self.CM = float(CM)
        self._def_hvec()
        return self.run()

    def change_hr(self, hr):
        self.hr = float(hr)
        return self.run()

    def change_A(self, A):
        self.A = float(A);  return self.run()

    def change_B(self, B):
        self.B = float(B);  return self.run()

    def change_C(self, C):
        self.C = float(C);  return self.run()

    def change_D(self, D):
        self.D = float(D);  return self.run()

    # ---------- CSV / calibração (opcional) ----------
    def add_data(self, path_csv: str):
        df = pd.read_csv(path_csv, dtype={"X": float, "Y": float})
        xr = pd.to_numeric(df["X"], errors="coerce").to_numpy()
        yr = pd.to_numeric(df["Y"], errors="coerce").to_numpy()
        m = np.isfinite(xr) & np.isfinite(yr)
        self.x_raw, self.y_raw = xr[m], yr[m]

        # posição absoluta de shoreline no CSV (Y==HTL)
        idx = np.argsort(self.y_raw)
        self.x_drift = float(np.interp(self.HTL, self.y_raw[idx], self.x_raw[idx]))

        # janela vertical HTL..DoC
        mask = (self.y_raw >= self.HTL) & (self.y_raw <= self.doc)
        self.x_obs = (self.x_raw[mask] - self.x_drift)
        self.y_obs = (self.y_raw[mask])
        self.y_obs_rel = self.y_obs - self.HTL
        self.data = True

        return self.calibrate()

    def calibrate(self):
        if not self.data:
            raise ValueError("Use add_data() antes de calibrar.")

        # Interpola X observado em função de h (=Y-HTL)
        idx = np.argsort(self.y_obs_rel)
        h_obs = self.y_obs_rel[idx]
        x_obs = self.x_obs[idx]
        x_obs_interp = np.interp(self.h, h_obs, x_obs, left=np.nan, right=np.nan)

        def model_x_of_h(A, B, C, D, hr):
            xo = ((hr + self.CM) / A) ** 1.5 - (hr / C) ** 1.5 + (B / (A ** 1.5)) * (hr + self.CM) ** 3 - (D / (C ** 1.5)) * hr ** 3
            mask1 = self.h <= (hr + self.CM)
            h1 = self.h[mask1]
            x1 = (h1 / A) ** 1.5 + B / (A ** 1.5) * h1 ** 3
            h2 = self.h[~mask1] - self.CM
            x2 = (h2 / C) ** 1.5 + D / (C ** 1.5) * h2 ** 3 + xo
            x_model = np.empty_like(self.h)
            x_model[mask1] = x1
            x_model[~mask1] = x2
            return x_model

        def residuals(plog):
            A, B, C, D = np.exp(plog[:4]); hr = plog[4]
            xm = model_x_of_h(A, B, C, D, hr)
            res = xm - x_obs_interp
            return res[np.isfinite(res)]

        p0 = np.array([np.log(max(self.A, 1e-5)),
                       np.log(max(self.B, 1e-8)),
                       np.log(max(self.C, 1e-5)),
                       np.log(max(self.D, 1e-8)),
                       self.hr])
        lb = [np.log(1e-5), np.log(1e-8), np.log(1e-5), np.log(1e-8), 0.0]
        ub = [np.log(10.0), np.log(1e2), np.log(10.0), np.log(1e2), float(self.h.max())]

        res = least_squares(residuals, p0, bounds=(lb, ub), loss="huber", f_scale=0.1, xtol=1e-8, ftol=1e-8, max_nfev=4000)
        self.A, self.B, self.C, self.D = np.exp(res.x[:4])
        self.hr = float(res.x[4])
        self._def_hvec()
        return self.run()

    def calibrate_segment_x(self, x0_abs: float):
        """
        Calibra A,B,C,D minimizando SSE em X no segmento [X0 → Xdoc] do CSV,
        ANCORANDO o modelo em (X0, Y0_csv). Retorna:
        x_full, y_full, (A,B,C,D,hr), X0, Y0, Xdoc
        onde (x_full,y_full) já estão projetados de HTL→DoC e transladados para
        passar por (X0,Y0).
        """
        if not self.data:
            raise RuntimeError("Sem CSV. Use add_data() antes da calibração.")

        X0 = float(x0_abs)
        Y0 = float(self._interp_y_at_x_csv(X0))
        Xdoc = float(self._x_doc_from_csv())

        # Observações no intervalo [X0, Xdoc]
        xr = np.asarray(self.x_raw, float)
        yr = np.asarray(self.y_raw, float)
        xmin, xmax = (X0, Xdoc) if X0 <= Xdoc else (Xdoc, X0)
        m_seg = (xr >= xmin) & (xr <= xmax)
        if not np.any(m_seg):
            raise RuntimeError("Não há pontos do CSV entre X0 e DoC para calibrar.")
        x_seg = xr[m_seg]
        y_seg = yr[m_seg]

        # y_model (grade vertical do Bernabeu) e função para interpolar x_model(y)
        # a partir de (A,B,C,D)
        def xmodel_from_params(params):
            A, B, C, D = params
            x_raw, x1_raw, x2_raw, h2 = bernabeu_mod(A, B, C, D, self.CM, self.h, self._def_xo())
            y_mod = self.h + self.HTL   # mesma grade do modelo
            # garantir ordem crescente de y para interp
            idx = np.argsort(y_mod)
            y_sorted = y_mod[idx]
            x_sorted = x_raw[idx]
            # x no Y0 para ancoragem
            x_at_Y0 = np.interp(Y0, y_sorted, x_sorted)
            shift = X0 - x_at_Y0
            # x previsto nos y do CSV (segmento)
            x_pred_seg = np.interp(y_seg, y_sorted, x_sorted) + shift
            return x_sorted + shift, y_sorted, x_pred_seg

        # Objetivo: SSE em X
        p0 = np.array([self.A, self.B, self.C, self.D], float)
        # bounds suaves em torno do estado atual (evita fugir demasiado)
        # ajuste se quiser mais liberdade
        bounds = [(1e-4, 5.0),   # A
                (1e-4, 1.0),   # B
                (1e-4, 1.0),   # C
                (0.0,   1.0)]  # D
        from scipy.optimize import minimize

        def sse(params):
            # penaliza fora de domínio
            if np.any(~np.isfinite(params)) or np.any(params <= 0) or params[3] < 0:
                return 1e30
            _, _, x_pred_seg = xmodel_from_params(params)
            res = x_pred_seg - x_seg
            return float(np.sum(res * res))

        res = minimize(sse, p0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-12})
        Ahat, Bhat, Chat, Dhat = (res.x if res.success else p0)

        # perfil completo HTL→DoC com *mesma* ancoragem (X0,Y0)
        x_sorted_shift, y_sorted, _ = xmodel_from_params([Ahat, Bhat, Chat, Dhat])

        # salvar parâmetros calibrados
        self.A, self.B, self.C, self.D = float(Ahat), float(Bhat), float(Chat), float(Dhat)

        # devolver no mesmo formato usado pelo test
        return x_sorted_shift, y_sorted, (self.A, self.B, self.C, self.D, self.hr), X0, Y0, Xdoc

    def _y_transition(self) -> float:
        # y absoluto no ponto de transição surf→shoaling
        # h = (hr + CM) (profundidade positiva); y = h + HTL
        return float(self.hr + self.CM + self.HTL)

    def _split_parts_from_full(self, x_full: np.ndarray, y_full: np.ndarray):
        """
        Gera (x_surf, y_surf) e (x_shoal, y_shoal) a partir do perfil completo,
        usando y_transition = hr+CM+HTL. Isso garante alinhamento perfeito entre
        as partes e o perfil concatenado.
        """
        y_tr = self._y_transition()
        m_surf  = (y_full <= y_tr)   # eixo invertido: 'menor' = mais raso
        m_shoal = ~m_surf

        x1 = x_full[m_surf].copy()
        y1 = y_full[m_surf].copy()
        x2 = x_full[m_shoal].copy()
        y2 = y_full[m_shoal].copy()
        return x1, y1, x2, y2

    # ---------- Acessores para perfis teóricos (surf/shoaling e completo) ----------
    def theoretical_parts(self, *, recompute: bool = True):
        """
        Retorna os dois ramos teóricos (surf e shoaling) alinhados ao estado atual.
        Parameters
        ----------
        recompute : bool
            Se True, roda self.run() para garantir que os arrays estejam atualizados.
        Returns
        -------
        (x_surf, y_surf, x_shoal, y_shoal)
        """
        if recompute or (self.x1 is None or self.x2 is None):
            self.run()
        return self.x1, self.y1, self.x2, self.y2

    def theoretical_full(self, *, recompute: bool = True):
        """
        Retorna o perfil completo teórico (surf+shoaling concatenado) alinhado ao estado atual.
        Returns
        -------
        (x_full, y_full)
        """
        if recompute or (self.x_full is None or self.y_full is None):
            return self.run()
        return self.x_full, self.y_full
    
    def _x_doc_from_csv(self) -> float:
        """
        Retorna o X absoluto do CSV onde Y cruza DoC.
        """
        if self.x_raw is None or self.y_raw is None:
            raise ValueError("Sem CSV carregado.")
        idx = np.argsort(self.y_raw)
        return float(np.interp(self.doc, self.y_raw[idx], self.x_raw[idx]))
    
    def metrics_segment(self, x_full, y_full, x0_abs: float):
        """
        RMSE/R² entre o perfil completo e o CSV, considerando somente
        o trecho X ∈ [X0, Xdoc], onde Xdoc é onde o CSV cruza DoC.
        """
        if self.x_raw is None or self.y_raw is None:
            return None, None

        # X de fechamento derivado do CSV
        try:
            x_doc = self._x_doc_from_csv()
        except Exception:
            return None, None

        xmin, xmax = (x0_abs, x_doc) if x0_abs <= x_doc else (x_doc, x0_abs)

        # recorta observações nesse intervalo
        m_csv = (np.isfinite(self.x_raw) & np.isfinite(self.y_raw) &
                 (self.x_raw >= xmin) & (self.x_raw <= xmax))
        if not np.any(m_csv):
            return None, None

        x_use = self.x_raw[m_csv]
        y_obs = self.y_raw[m_csv]

        # recorta perfil do modelo no mesmo domínio de X
        m_mod = (np.isfinite(x_full) & np.isfinite(y_full) &
                 (x_full >= xmin) & (x_full <= xmax))
        if not np.any(m_mod):
            return None, None

        # interpola y_model(x) no grid de observação
        y_mod = np.interp(x_use, x_full[m_mod], y_full[m_mod])

        resid = y_obs - y_mod
        rmse = float(np.sqrt(np.mean(resid**2)))
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y_obs - np.mean(y_obs))**2))
        r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else None
        return rmse, r2
    
    def x_at_transition(self, x_full, y_full) -> float:
        """Retorna X onde y == (hr+CM+HTL) por interpolação."""
        y_tr = self._y_transition()
        # Garante monotonia em X para a interpolação:
        idx = np.argsort(x_full)
        x_sorted = x_full[idx]
        y_sorted = y_full[idx]
        return float(np.interp(y_tr, y_sorted, x_sorted))
    
    # --- helpers CSV (mesma lógica usada no Dean) --------------------------------
    def _interp_y_at_x_csv(self, xq: float) -> float:
        """Interpolar (com extrapolação linear nas bordas) Y_csv em X=xq."""
        if not self.data or self.x_raw is None or self.y_raw is None:
            raise RuntimeError("CSV não carregado. Use add_data() antes.")
        xr = np.asarray(self.x_raw, float)
        yr = np.asarray(self.y_raw, float)
        if xr.size < 2:
            return float(yr[0])
        if xq <= xr.min():
            i0, i1 = 0, 1
        elif xq >= xr.max():
            i0, i1 = -2, -1
        else:
            i1 = int(np.searchsorted(xr, xq))
            i0 = max(0, i1 - 1)
        x0, x1 = float(xr[i0]), float(xr[i1])
        y0, y1 = float(yr[i0]), float(yr[i1])
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * ((xq - x0) / (x1 - x0))

    def _x_doc_from_csv(self) -> float:
        """X (absoluto) onde o CSV cruza Y=DoC (interpola ou extrapola borda)."""
        if not self.data or self.x_raw is None or self.y_raw is None:
            raise RuntimeError("CSV não carregado. Use add_data() antes.")
        xr = np.asarray(self.x_raw, float)
        yr = np.asarray(self.y_raw, float)
        dif = yr - self.doc
        s = np.sign(dif)
        cross = np.where(s[:-1] * s[1:] <= 0)[0]
        if cross.size:
            i = int(cross[0])
            x0, x1 = xr[i], xr[i+1]
            y0, y1 = yr[i], yr[i+1]
            if y1 == y0:
                return float(x0)
            return float(x0 + (self.doc - y0) * (x1 - x0) / (y1 - y0))
        # sem cruzamento → extrapolar borda mais próxima
        k = int(np.argmin(np.abs(dif)))
        if k == 0:
            x0, x1 = xr[0], xr[1]; y0, y1 = yr[0], yr[1]
        else:
            x0, x1 = xr[-2], xr[-1]; y0, y1 = yr[-2], yr[-1]
        if y1 == y0:
            return float(x0)
        return float(x0 + (self.doc - y0) * (x1 - x0) / (y1 - y0))

        
    

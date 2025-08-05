import sys, os
# inclui ../src no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', 'src')))
from IHSetBernabeu.cal_bernabeu import cal_Bernabeu

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class BernabeuTest:

    def __init__(self):
        self.HTL  = -0.25 # High tide level (vertical displacement)
        self.Hs50 = 1.0 # Mean significant wave height
        self.Tp50 = 8.0 # Average peak wave period
        self.D50  = 0.4 # Median sediment grain size, in millimeters.
        self.doc  = 2.75 # Beach profile - depth of closure
        self.CM   = 2.0 # depth of start of shoaling zone
        self.hr   = 1.0 # depth of the inflection point between surf and shoaling
        self.csv  = "XY_PuertoChiquito_clean.csv"

        # 1. Instantiate model
        self.model = cal_Bernabeu(self.CM, self.Hs50, self.D50, self.Tp50,
                            self.doc, self.hr, self.HTL)

    def plot(self):
        # 2 - Generates theoretical profile without calibration from D50
        x_theo, y_theo = self.model.from_D50(self.D50)

        noise = np.random.normal(loc=0.0, scale=0.2, size=x_theo.shape)
        synthetic = pd.DataFrame({'X': x_theo, 'Y': y_theo + noise})
        synthetic.to_csv('Bernabeu_profile.csv', index=False)

        # 3 - Defines the color settings for painting beach profiles
        LIGHTSAND = "#f4dcb8"   # areia clara
        DARKSAND  = "#d6b07a"   # areia escura
        WATER     = "#a6cee3"   # água

        # 4 - Sets the plot limits of the graph
        x_min, x_max = x_theo.min(), x_theo.max()
        y_bottom = y_theo.max()
        y_top = self.HTL - 2.0  # 2 m above sea level (SL), Y axes inverted

        #print(f"x_min = {x_min}, x_max = {x_max}, y_bottom = {y_bottom}, y_top = {y_top}")
        
        # 5 - Plot the graph comparing the synthetic profile with a measured one
        # 5.1. Fill water color (between HTL and y_bottom)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.fill_between(
            [x_min, x_max],
            self.model.HTL, y_bottom,
            color=WATER, alpha=0.7, zorder=1
        )

        # 5.2. Fill dark sand under theoretical profile (y_theo)
        ax.fill_between(
            x_theo, y_theo, y_bottom,
            color=DARKSAND, alpha=0.7, zorder=3
        )

        # 5.3. Waterline (HTL)
        ax.plot(
            [x_min, x_max],
            [self.model.HTL, self.model.HTL],
            color="blue", lw=1.5,
            label="High Tide (HTL))", zorder=0
        )

        # 5.4. Theorethical Bernabeu profile line
        ax.plot(
            x_theo, y_theo,
            '--', linewidth=2,
            color='red',
            label='Bernabeu Profile', zorder=5
        )

        # 5.7. Theoretical Bernabeu profile graphic final adjustments ---
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_top, y_bottom)
        ax.set_xlabel("Cross‑Shore Distance - X [m]")
        ax.set_ylabel("Elevation / Depth - Y [m]")
        ax.invert_yaxis()                 # emerged at the top, deep below
        ax.grid(True, linestyle=':', lw=0.5)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        plt.title('Bernabeu Profile from D50')
        plt.tight_layout()
        plt.show()

        
        """
        ## 6. Other bernabeu theorethical profiles controled by other Bernabeu parameters

        # 6.1 Changes in D50
        (x_theo, y_theo) = self.model.from_Hs50(2.0)

        plt.plot(x_theo, y_theo, label='Bernabeu Profile from Hs50')
        plt.title('Bernabeu Profile from Hs50')
        plt.show()

        # 6.2 Changes in Tp50
        (x_theo, y_theo) = self.model.from_Tp50(10.0)

        plt.plot(x_theo, y_theo, label='Bernabeu Profile from Tp50')
        plt.title('Bernabeu Profile from Tp50')
        plt.show()

        # 6.3 Changes in CM
        (x_theo, y_theo) = self.model.change_CM(3.0)

        plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed CM')
        plt.title('Bernabeu Profile with changed CM')
        plt.show()

        6.4 Changes in doc
        (x_theo, y_theo) = self.model.change_doc(12.0)

        plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed doc')
        plt.title('Bernabeu Profile with changed doc')
        plt.show()

        # 6.5 Changes in HTL
        (x_theo, y_theo) = self.model.change_HTL(-5.0)

        plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed HTL')
        plt.title('Bernabeu Profile with changed HTL')
        plt.show()

        # 6.6 Changes in hr
        (x_theo, y_theo) = self.model.change_hr(2.0)
        
        plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed hr')
        plt.title('Bernabeu Profile with changed hr')
        plt.show()

        # 6.7 Changes in A parameter
        (x_theo, y_theo) = self.model.change_A(0.1)

        plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed A')
        plt.title('Bernabeu Profile with changed A')
        plt.show()

        # 6.7 Changes in B parameter
        (x_theo, y_theo) = self.model.change_B(0.01)

        plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed B')
        plt.title('Bernabeu Profile with changed B')
        plt.show()

        # 6.7 Changes in C parameter
        (x_theo, y_theo) = model.change_C(0.07)

        plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed C')
        plt.title('Bernabeu Profile with changed C')
        plt.show()
        
        # 6.7 Changes in D parameter
        (x_theo, y_theo) = self.model.change_D(0.001)

        plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed D')
        plt.title('Bernabeu Profile with changed D')
        plt.show()
        
        """

        # 8 - Plot the calibrated Bernabeu Profile

        # 8.0. Load the CSV profile - The ground truth
        df_field = pd.read_csv(self.csv, dtype={'X': float, 'Y': float})
        self.x_obs = df_field['X'].values
        self.y_obs = pd.to_numeric(df_field['Y'], errors='coerce').values
        # ajusta sinal para elevação positiva acima da água
        if np.mean(self.y_obs) < 0:
            self.y_obs = -self.y_obs

        # 8.1. Alternative noise profile
        #(self.x_cal, self.y_cal) = self.model.add_data('Bernabeu_profile.csv')
        (x_cal, y_cal) = self.model.add_data(self.csv)

        # 8.2. Print the Bernabeu calibration values
        print(f"Ar = {self.model.Ar}, B = {self.model.B}, C = {self.model.C}, D = {self.model.D}, CM = {self.model.CM}, hr = {self.model.hr}")

        # 8.3. Fill waterline (between HTL e y_bottom)
        fig, ax     = plt.subplots(figsize=(8,5))

        all_x_cal               = np.concatenate([self.x_obs, x_cal])
        all_y_cal               = np.concatenate([self.y_obs, y_cal])
        x_cal_min, x_cal_max    = all_x_cal.min(), all_x_cal.max()
        y_cal_bottom            = all_y_cal.max()   # highest deep
        y_top                   = self.model.HTL - 2

        ax.fill_between(
            [x_cal_min, x_cal_max],
            self.model.HTL, y_cal_bottom,
            color=WATER, alpha=0.7, zorder=1
        )

        # 8.4. Fill light sand under CSV profile (y_field)
        ax.fill_between(
            self.model.x_raw, self.model.y_raw, y_cal_bottom,
            color=LIGHTSAND, zorder=2
        )

        # 8.5. Fill dark sand under Barnabeu calibrated profile (y_theo)
        ax.fill_between(
            x_cal, y_cal, y_cal_bottom,
            color=DARKSAND, alpha=0.7, zorder=3
        )

        # 8.6. Waterline (HTL)
        ax.plot(
            [x_cal_min, x_cal_max],
            [self.model.HTL, self.model.HTL],
            color="blue", lw=1.5,
            label="High Tide (HTL))", zorder=0
        )

        # 8.7. CSV profile line
        ax.plot(
            self.model.x_raw, self.model.y_raw,
            '-', linewidth=2,
            color='black',
            label='Observed Data', zorder=4
        )

        # 8.8. Bernabeu calibrated line
        ax.plot(
            x_cal, y_cal,
            '--', linewidth=2,
            color='red',
            label='Bernabeu Profile', zorder=7
        )

        # 8.9. Segments lines
        #ax.plot(self.model.x1_full, self.model.y1_full, ':', color='yellow', lw=1.8, label='Surf zone',   zorder=5)
        #ax.plot(self.model.x2_full, self.model.y2_full, ':', color='green',  lw=1.8, label='Shoaling zone',zorder=6)

        # 8.10. Final adjustments ---
        ax.set_xlim(x_cal_min, x_cal_max)
        ax.set_ylim(y_top, y_cal_bottom)
        ax.set_xlabel("Cross‑Shore Distance - X [m]")
        ax.set_ylabel("Elevation / Depth - Y [m]")
        ax.invert_yaxis()
        ax.grid(True, linestyle=':', lw=0.5)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        plt.title('Bernabeu Profile with added data')
        plt.tight_layout()
        plt.show()

# Run if executed as script
if __name__ == "__main__":
    tst = BernabeuTest()
    tst.plot()
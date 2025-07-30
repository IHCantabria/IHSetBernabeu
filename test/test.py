import sys, os
# inclui ../src no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', 'src')))
from IHSetBernabeu.cal_bernabeu import cal_Bernabeu

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


HTL  = -2.0 # High tide level (vertical displacement)
Hs50 = 1.0 # Mean significant wave height
Tp50 = 8.0 # Average peak wave period
D50  = 0.4 # Median sediment grain size, in millimeters.
doc  = 8.0 # Beach profile - depth of closure
CM   = 4.0 # depth of start of shoaling zone
hr   = 1.5 # depth of the inflection point between surf and shoaling

# Instantiate model
model = cal_Bernabeu(CM=CM, Hs50=Hs50, D50=D50, Tp50=Tp50,
                     doc=doc, hr=hr, HTL=HTL)

# 1 - Load the CSV profile - The ground truth
df_field = pd.read_csv('XY_PuertoChiquito_clean.csv', dtype={'X': float, 'Y': float})
x_obs = df_field['X'].values
y_obs = pd.to_numeric(df_field['Y'], errors='coerce').values
# ajusta sinal para elevação positiva acima da água
if np.mean(y_obs) < 0:
    y_obs = -y_obs

x_raw = x_obs
y_raw = y_obs

# 2 - Generates theoretical profile without calibration from D50
x_theo, y_theo = model.from_D50(D50)

noise = np.random.normal(loc=0.0, scale=0.2, size=x_theo.shape)
synthetic = pd.DataFrame({'X': x_theo, 'Y': y_theo + noise})
synthetic.to_csv('Bernabeu_profile.csv', index=False)

# 3 - Defines the color settings for painting beach profiles
LIGHTSAND = "#f4dcb8"   # areia clara
DARKSAND  = "#d6b07a"   # areia escura
WATER     = "#a6cee3"   # água

# 4 - Sets the plot limits of the graph
all_x = np.concatenate([x_obs, x_theo])
all_y = np.concatenate([y_obs, y_theo])
x_min, x_max = all_x.min(), all_x.max()
y_bottom    = all_y.max()   # highest deep
y_top       = model.HTL - 2 # two metters above HTL

# 5 - Plot the graph comparing the synthetic profile with a measured one
# 5.1. Fill water color (between HTL and y_bottom)
fig, ax = plt.subplots(figsize=(8,5))
ax.fill_between(
    [x_min, x_max],
    model.HTL, y_bottom,
    color=WATER, alpha=0.7, zorder=1
)

# 5.2. Fill light sand under CSV profile (y_field)
#ax.fill_between(
#    x_field, y_field, y_bottom,
#    color=LIGHTSAND, zorder=2
#)

# 5.3. Fill dark sand under theoretical profile (y_theo)
ax.fill_between(
    x_theo, y_theo, y_bottom,
    color=DARKSAND, alpha=0.7, zorder=3
)

# 5.4. Waterline (HTL)
ax.plot(
    [x_min, x_max],
    [model.HTL, model.HTL],
    color="blue", lw=1.5,
    label="High Tide (HTL))", zorder=0
)

# 5.5. True ground profile line (no noise)
#ax.plot(
#    x_field, y_field,
#    '-', linewidth=2,
#    color='black',
#    label='True Profile - CSV', zorder=4
#)

# 5.6 Theorethical Bernabeu profile line
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
plt.plot(x_theo, y_theo, '-.', label='Bernabeu Profile')
plt.plot(x_field, y_field, '-', linewidth=2, label='Perfil verdadeiro (sem ruído)', color='black')
plt.title('Bernabeu Profile from D50')
plt.show()

(x_theo, y_theo) = model.from_Hs50(2.0)

plt.plot(x_theo, y_theo, label='Bernabeu Profile from Hs50')
plt.title('Bernabeu Profile from Hs50')
plt.show()

(x_theo, y_theo) = model.from_Tp50(10.0)

plt.plot(x_theo, y_theo, label='Bernabeu Profile from Tp50')
plt.title('Bernabeu Profile from Tp50')
plt.show()

(x_theo, y_theo) = model.change_CM(3.0)

plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed CM')
plt.title('Bernabeu Profile with changed CM')
plt.show()

(x_theo, y_theo) = model.change_doc(12.0)

plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed doc')
plt.title('Bernabeu Profile with changed doc')
plt.show()

(x_theo, y_theo) = model.change_HTL(-5.0)

plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed HTL')
plt.title('Bernabeu Profile with changed HTL')
plt.show()


(x_theo, y_theo) = model.change_hr(2.0)
plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed hr')
plt.title('Bernabeu Profile with changed hr')
plt.show()

(x_theo, y_theo) = model.change_A(0.1)

plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed A')
plt.title('Bernabeu Profile with changed A')
plt.show()

(x_theo, y_theo) = model.change_B(0.01)

plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed B')
plt.title('Bernabeu Profile with changed B')
plt.show()

(x_theo, y_theo) = model.change_C(0.07)
plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed C')
plt.title('Bernabeu Profile with changed C')
plt.show()

(x_theo, y_theo) = model.change_D(0.001)
plt.plot(x_theo, y_theo, label='Bernabeu Profile with changed D')
plt.title('Bernabeu Profile with changed D')
plt.show()

#(x_theo,y_theo) = model.add_data(r'XY_PuertoChiquito_clean.csv')
"""

# 8 - Plot the calibrated Bernabeu Profile
# 8.1. Alternative noise profile
(x_cal,y_cal) = model.add_data('Bernabeu_profile.csv')
#(x_cal,y_cal) = model.add_data('XY_PuertoChiquito_clean.csv')

# 8.2. Print the Bernabeu calibration values
print(f"Ar = {model.Ar}, B = {model.B}, C = {model.C}, D = {model.D}, CM = {model.CM}, hr = {model.hr}")

# 8.3. Fill waterline (between HTL e y_bottom)
fig, ax     = plt.subplots(figsize=(8,5))

all_x_cal               = np.concatenate([x_raw, x_cal])
all_y_cal               = np.concatenate([y_raw, y_cal])
x_min_cal, x_max_cal    = all_x_cal.min(), all_x_cal.max()
y_bottom_cal            = all_y_cal.max()   # highest deep
y_top                   = model.HTL - 2

ax.fill_between(
    [x_min_cal, x_max_cal],
    model.HTL, y_bottom_cal,
    color=WATER, alpha=0.7, zorder=1
)

# 8.4. Fill light sand under CSV profile (y_field)
ax.fill_between(
    model.x_obs, model.y_obs, y_bottom_cal,
    color=LIGHTSAND, zorder=2
)

# 8.5. Fill dark sand under Barnabeu calibrated profile (y_theo)
ax.fill_between(
    x_cal, y_cal, y_bottom_cal,
    color=DARKSAND, alpha=0.7, zorder=3
)

# 8.6. Waterline (HTL)
ax.plot(
    [x_min_cal, x_max_cal],
    [model.HTL, model.HTL],
    color="blue", lw=1.5,
    label="High Tide (HTL))", zorder=0
)

# 8.7. CSV profile line
ax.plot(
    model.x_obs, model.y_obs,
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
#ax.plot(model.x1_full, model.y1_full, ':', color='yellow', lw=1.8, label='Surf zone',   zorder=5)
#ax.plot(model.x2_full, model.y2_full, ':', color='green',  lw=1.8, label='Shoaling zone',zorder=6)

# 8.10. Final adjustments ---
ax.set_xlim(x_min_cal, x_max_cal)
ax.set_ylim(y_top, y_bottom_cal)
ax.set_xlabel("Cross‑Shore Distance - X [m]")
ax.set_ylabel("Elevation / Depth - Y [m]")
ax.invert_yaxis()
ax.grid(True, linestyle=':', lw=0.5)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
plt.title('Bernabeu Profile with added data')
plt.tight_layout()
plt.show()

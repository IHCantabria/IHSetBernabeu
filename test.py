from src.IHSetBernabeu.cal_bernabeu_r2 import cal_Bernabeu
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

model = cal_Bernabeu(
    HTL=-2.0, # High tide level (vertical displacement)
    Hs50=1.0, # Mean significant wave height
    Tp50=8.0, # Average peak wave period
    D50=0.4, # Median sediment grain size, in millimeters.
    doc=10.0, # Beach profile - depth of closure
    CM=3.0, # depth of start of shoaling zone
    hr=1.5 # profundidade do ponto de inflexão entre surf e shoaling
)

# 1 - Carrega o perfil “verdadeiro” já existente (sem ruído)
df_true = pd.read_csv('XY_PuertoChiquito_clean.csv')
x_true = df_true['X'].values
y_true = df_true['Y'].values

# 2 - gera um perfil sintético de Dean a partir do D50
(x, y) = model.from_D50(0.3)

# - Adiciona ruído ao perfil sintético gerado a partir do D50
noise = np.random.normal(-0.2, 0.2, x.shape)

df = pd.DataFrame({
    'X': x,
    'Y': y + noise
})

# 3 - Salva o perfil sintético com ruído em um CSV
df.to_csv('Bernabeu_profile.csv', index=False)

# 4 - Define as configurações de cores para pintar os perfis de praia
LIGHTSAND = "#f4dcb8"   # areia clara
DARKSAND  = "#d6b07a"   # areia escura
WATER     = "#a6cee3"   # água

# 5 - Define os limites de plot do gráfico
all_x = np.concatenate([x_true, x])
all_y = np.concatenate([y_true, y])
x_min, x_max = all_x.min(), all_x.max()
y_bottom    = all_y.max()   # maior profundidade
y_top       = model.HTL - 2 # um metro acima do HTL

# 6 - Plota o gráfico comparando o pefil sintético com um medido
# 6.1. Pinta a água (entre HTL e y_bottom)
fig, ax = plt.subplots(figsize=(8,5))
ax.fill_between(
    [x_min, x_max],
    model.HTL, y_bottom,
    color=WATER, alpha=0.7, zorder=1
)

# 6.2. pinta a areia clara sob o perfil “verdadeiro” (y_true)
#ax.fill_between(
#    x_true, y_true, y_bottom,
#    color=LIGHTSAND, zorder=2
#)

# 6.3. pinta a areia escura sob o perfil calibrado (y_cal)
ax.fill_between(
    x, y, y_bottom,
    color=DARKSAND, alpha=0.7, zorder=3
)

# 6.4. linha do nível da água (HTL)
ax.plot(
    [x_min, x_max],
    [model.HTL, model.HTL],
    color="blue", lw=1.5,
    label="High Tide (HTL))", zorder=0
)

# 6.5. perfil verdadeiro (sem ruído)
#ax.plot(
#    x_true, y_true,
#    '-', linewidth=2,
#    color='black',
#    label='True Profile - CSV', zorder=4
#)

# 6.6. perfil calibrado (Bernabeu final)
ax.plot(
    x, y,
    '--', linewidth=2,
    color='red',
    label='Bernabeu Profile', zorder=5
)

# 6.7. formatação final ---
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_top, y_bottom)
ax.set_xlabel("Cross‑Shore Distance - X [m]")
ax.set_ylabel("Elevation / Depth - Y [m]")
ax.invert_yaxis()                 # emerso no topo, profundo embaixo
ax.grid(True, linestyle=':', lw=0.5)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
plt.title('Bernabeu Profile from D50')
plt.tight_layout()
plt.show()

"""
plt.plot(x, y, '-.', label='Bernabeu Profile')
plt.plot(x_true, y_true, '-', linewidth=2, label='Perfil verdadeiro (sem ruído)', color='black')
plt.title('Bernabeu Profile from D50')
plt.show()

# 7 - Plota uma série de variações baseadas em variações dos inputs de entrada do modelo
(x, y) = model.from_Hs50(2.0)

plt.plot(x, y, label='Bernabeu Profile from Hs50')
plt.title('Bernabeu Profile from Hs50')
plt.show()

(x, y) = model.from_Tp50(10.0)

plt.plot(x, y, label='Bernabeu Profile from Tp50')
plt.title('Bernabeu Profile from Tp50')
plt.show()

(x, y) = model.change_CM(3.0)

plt.plot(x, y, label='Bernabeu Profile with changed CM')
plt.title('Bernabeu Profile with changed CM')
plt.show()

(x, y) = model.change_doc(12.0)

plt.plot(x, y, label='Bernabeu Profile with changed doc')
plt.title('Bernabeu Profile with changed doc')
plt.show()

(x, y) = model.change_HTL(-5.0)

plt.plot(x, y, label='Bernabeu Profile with changed HTL')
plt.title('Bernabeu Profile with changed HTL')
plt.show()


(x, y) = model.change_hr(2.0)
plt.plot(x, y, label='Bernabeu Profile with changed hr')
plt.title('Bernabeu Profile with changed hr')
plt.show()

(x, y) = model.change_A(0.1)

plt.plot(x, y, label='Bernabeu Profile with changed A')
plt.title('Bernabeu Profile with changed A')
plt.show()

(x, y) = model.change_B(0.01)

plt.plot(x, y, label='Bernabeu Profile with changed B')
plt.title('Bernabeu Profile with changed B')
plt.show()

(x, y) = model.change_C(0.07)
plt.plot(x, y, label='Bernabeu Profile with changed C')
plt.title('Bernabeu Profile with changed C')
plt.show()

(x, y) = model.change_D(0.001)
plt.plot(x, y, label='Bernabeu Profile with changed D')
plt.title('Bernabeu Profile with changed D')
plt.show()

#(x,y) = model.add_data(r'XY_PuertoChiquito_clean.csv')
"""

# 8 - Plota o perfil calibrado de Bernabeu
model = cal_Bernabeu(
    HTL=-2.0, # High tide level (vertical displacement)
    Hs50=1.0, # Mean significant wave height
    Tp50=8.0, # Average peak wave period
    D50=0.4, # Median sediment grain size, in millimeters.
    doc=10.0, # Beach profile - depth of closure
    CM=3.0, # depth of start of shoaling zone
    hr=1.5 # profundidade do ponto de inflexão entre surf e shoaling
)

#(x,y) = model.add_data('Bernabeu_profile.csv')
(x,y) = model.add_data('XY_PuertoChiquito_clean.csv')

print(f"Ar = {model.Ar}, B = {model.B}, C = {model.C}, D = {model.D}, CM = {model.CM}, hr = {model.hr}")

# 9 - Define os limites de plot do gráfico
all_x = np.concatenate([model.x_obs, x])
all_y = np.concatenate([model.y_obs, y])
x_min, x_max = all_x.min(), all_x.max()
y_bottom    = all_y.max()   # maior profundidade
y_top       = model.HTL - 2 # 2 metros acima do HTL

# 10 - Plota o gráfico comparando o pefil sintético com um medido
# 10.1. Pinta a água (entre HTL e y_bottom)
fig, ax = plt.subplots(figsize=(8,5))
ax.fill_between(
    [x_min, x_max],
    model.HTL, y_bottom,
    color=WATER, alpha=0.7, zorder=1
)

# 10.2. pinta a areia clara sob o perfil “verdadeiro” (y_true)
ax.fill_between(
    model.x_obs, model.y_obs, y_bottom,
    color=LIGHTSAND, zorder=2
)

# 10.3. pinta a areia escura sob o perfil calibrado (y_cal)
ax.fill_between(
    x, y, y_bottom,
    color=DARKSAND, alpha=0.7, zorder=3
)

# 10.4. linha do nível da água (HTL)
ax.plot(
    [x_min, x_max],
    [model.HTL, model.HTL],
    color="blue", lw=1.5,
    label="High Tide (HTL))", zorder=0
)

# 6.5. perfil verdadeiro (sem ruído)
ax.plot(
    model.x_obs, model.y_obs,
    '-', linewidth=2,
    color='black',
    label='Observed Data', zorder=4
)

# 6.6. perfil calibrado (Bernabeu final)
ax.plot(
    x, y,
    '--', linewidth=2,
    color='red',
    label='Bernabeu Profile', zorder=7
)

ax.plot(model.x1, y, '--', linewidth=2, color='yellow', label='Segment 1', zorder=5)
ax.plot(model.x2, model.y2, '--', linewidth=2, color='green', label='Segment 2', zorder=6)

# 6.7. formatação final ---
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_top, y_bottom)
ax.set_xlabel("Cross‑Shore Distance - X [m]")
ax.set_ylabel("Elevation / Depth - Y [m]")
ax.invert_yaxis()                 # emerso no topo, profundo embaixo
ax.grid(True, linestyle=':', lw=0.5)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
plt.title('Bernabeu Profile with added data')
plt.tight_layout()
plt.show()

"""
plt.plot(model.x_obs, model.y_obs, '.', label='Observed Data', markersize=2, color='black')
plt.plot(model.x1, y, label='Segment 1', linewidth=2, color='red')
plt.plot(model.x2, model.y2, label='Segment 2', linewidth=2, color='green')
plt.plot(x, y, label='Bernabeu Profile with added data', linewidth=1, color='blue', linestyle='--')

plt.title('Bernabeu Profile with added data')
plt.show()
"""

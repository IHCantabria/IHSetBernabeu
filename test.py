from src.IHSetBernabeu.cal_bernabeu import cal_Bernabeu
import matplotlib.pyplot as plt

model = cal_Bernabeu(
    CM=4.0,
    Hs50=1.0,
    hr = 1.5,
    D50=0.3,
    Tp50=8.0,
    doc=10.0,
    HTL=-4.0
)

(x, y) = model.from_D50(0.3)

import pandas as pd
import numpy as np

noise = np.random.normal(-0.2, 0.2, x.shape)


df = pd.DataFrame({
    'X': x,
    'Y': y + noise
})
df.to_csv('Bernabeu_profile.csv', index=False)


# # plt.plot(x1, y, label='Segment 1', linewidth=1, color='red')
# # plt.plot(x2, y2, label='Segment 2', linewidth=1, color='black')
# # plt.plot(xx, y, label='Segment 2 Interpolated', linewidth=1, color='blue')
plt.plot(x, y, '-.', label='Bernabeu Profile')
plt.title('Bernabeu Profile from D50')
plt.show()

# (x, y) = model.from_Hs50(2.0)

# plt.plot(x, y, label='Bernabeu Profile from Hs50')
# plt.title('Bernabeu Profile from Hs50')
# plt.show()

# (x, y) = model.from_Tp50(10.0)

# plt.plot(x, y, label='Bernabeu Profile from Tp50')
# plt.title('Bernabeu Profile from Tp50')
# plt.show()


# (x, y) = model.change_CM(3.0)

# plt.plot(x, y, label='Bernabeu Profile with changed CM')
# plt.title('Bernabeu Profile with changed CM')
# plt.show()

# # (x, y) = model.change_doc(12.0)

# plt.plot(x, y, label='Bernabeu Profile with changed doc')
# plt.title('Bernabeu Profile with changed doc')
# plt.show()

# (x, y) = model.change_HTL(-5.0)

# plt.plot(x, y, label='Bernabeu Profile with changed HTL')
# plt.title('Bernabeu Profile with changed HTL')
# plt.show()


# (x, y) = model.change_hr(2.0)
# plt.plot(x, y, label='Bernabeu Profile with changed hr')
# plt.title('Bernabeu Profile with changed hr')
# plt.show()

# (x, y) = model.change_A(0.1)

# plt.plot(x, y, label='Bernabeu Profile with changed A')
# plt.title('Bernabeu Profile with changed A')
# plt.show()

# (x, y) = model.change_B(0.01)

# plt.plot(x, y, label='Bernabeu Profile with changed B')
# plt.title('Bernabeu Profile with changed B')
# plt.show()

# (x, y) = model.change_C(0.07)
# plt.plot(x, y, label='Bernabeu Profile with changed C')
# plt.title('Bernabeu Profile with changed C')
# plt.show()

# (x, y) = model.change_D(0.001)
# plt.plot(x, y, label='Bernabeu Profile with changed D')
# plt.title('Bernabeu Profile with changed D')
# plt.show()

# (x,y) = model.add_data(r'C:\Users\freitasl\Documents\IH_SET_Repos\IHSetBernabeu\XY_PuertoChiquito_clean.csv')


model = cal_Bernabeu(
    CM=4.0,
    Hs50=1.0,
    hr = 1.5,
    D50=0.2,
    Tp50=8.0,
    doc=10.0,
    HTL=-4.0
)

(x,y) = model.add_data('Bernabeu_profile.csv')

print(f"Ar = {model.Ar}, B = {model.B}, C = {model.C}, D = {model.D}, CM = {model.CM}, hr = {model.hr}")


plt.plot(model.x_obs, model.y_obs, '.', label='Observed Data', markersize=2, color='black')
plt.plot(model.x1, y, label='Segment 1', linewidth=2, color='red')
plt.plot(model.x2, model.y2, label='Segment 2', linewidth=2, color='green')
plt.plot(x, y, label='Bernabeu Profile with added data', linewidth=1, color='blue', linestyle='--')


plt.title('Bernabeu Profile with added data')
plt.show()


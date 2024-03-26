from IHSetBernabeu import IHSetBernabeu
import os
import matplotlib.pyplot as plt

wrkDir = os.getcwd()
model = IHSetBernabeu.cal_Bernabeu(wrkDir+'/data/')
self = IHSetBernabeu.Bernabeu(model.calibrate())

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.weight': 'bold'})
font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}

hk = []
hk.append(plt.plot(self.dp, self.zp, '--k')[0])
hk.append(plt.plot(self.dp, self.hmi, linewidth=2)[0])
plt.show()
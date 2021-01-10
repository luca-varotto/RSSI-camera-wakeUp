import matplotlib.pyplot as plt
import numpy as np 
from scipy.stats import norm, lognorm

sigma = 3
sigma_d = 1
d = 5
n=2

fig =plt.figure()
ax= plt.gca()
x = np.arange(-15, 15, 0.1)
norm_pdf = norm.pdf(x,0,sigma)
lognorm_pdf = lognorm.pdf(x,s=sigma_d/d,scale=np.exp(1)* np.log(10)/(10*n) )
plt.plot(x,(norm_pdf+lognorm_pdf)/max(norm_pdf+lognorm_pdf),label='LN+N')
plt.plot(x,lognorm_pdf/max(norm_pdf+lognorm_pdf),label='LN',linestyle=':')
plt.plot(x,norm_pdf/max(norm_pdf+lognorm_pdf),label='N',linestyle=':')
plt.legend()
plt.show()
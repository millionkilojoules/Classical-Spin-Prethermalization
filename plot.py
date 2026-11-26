import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FixedLocator
from functions import Omega_list,J,linear_fitting



data_1=np.load('data/cycles.npz')
cycles=data_1['cycles']

data_2=np.load('data/result.npz')
Q_all,delta_Q_all,S_z=data_2['Q_all'],data_2['delta_Q_all'],data_2['S_z']


#Format
mpl.rcParams.update({
	"text.usetex": True,"text.latex.preamble": r"\usepackage{amsmath}",
	"font.family": "serif",
	"font.size": 10,"axes.labelsize": 12,"xtick.labelsize": 10,"ytick.labelsize": 10,"legend.fontsize": 9,
	"axes.linewidth": 0.8,
	"xtick.direction": "in","ytick.direction": "in","xtick.top": True,"ytick.right": True,
	"lines.linewidth": 1.2,
})


plateau=np.zeros(len(Omega_list)-1)


#Q and δQ
fig1,ax1=plt.subplots(2,1,figsize=(8,8),sharex=True,gridspec_kw={'hspace':0})
ax1 = ax1.ravel()

max_exp = int(np.floor(np.log10(cycles[-1])))
ticks = 10.0 ** np.arange(1, max_exp + 1, 2)

for a in ax1:
	a.xaxis.set_major_locator(FixedLocator(ticks))
	a.grid(True, which='major', axis='x',linestyle='--', alpha=0.5)
	a.grid(True, which='major', axis='y',linestyle='--', alpha=0.5)

for k,Omega in enumerate(Omega_list[:-1]):
	ax1[0].semilogx(cycles[1:],Q_all[k,1:],label=fr"$\Omega/J={Omega/J:.2f}$")
	ax1[1].semilogx(cycles[1:], delta_Q_all[k, 1:], label=fr"$\Omega/J={Omega / J:.2f}$")

	plt.show(block=False)
	point=plt.ginput(1,timeout=0)
	plateau[k]=point[0][0]

k,Omega=4,4.6
ax1[0].semilogx(cycles[1:],Q_all[k,1:],label=fr"$\Omega/J={Omega/J:.2f}$")
ax1[1].semilogx(cycles[1:], delta_Q_all[k, 1:], label=fr"$\Omega/J={Omega / J:.2f}$")

for k, Omega in enumerate(Omega_list[:-1]):
	ax1[0].axvline(plateau[k], color='black', linestyle='--', alpha=1)
	ax1[1].axvline(plateau[k], color='black', linestyle='--', alpha=1)
#if every δQ gets a peak,there's no need to Omega_list[:-1]



ax1[0].set_xlabel('$l$')
ax1[0].set_ylabel('$<Q(lT)>$')
ax1[0].legend()

ax1[1].set_xlabel('$l$')
ax1[1].set_ylabel(r'$<\delta Q(lT)>$')
ax1[1].legend()

fig1.suptitle('Noise-Averaged Energy and Energy Variance')


#τ-Ω/J relationship
fig2,ax2=plt.subplots(figsize=(5,5))

ln_plateau=np.log(plateau)
y_fit,r_squared,k,b=linear_fitting(Omega_list[:-1],ln_plateau)
plateau_fit=np.e**(k*Omega_list[:-1]+b)

ax2.semilogy(Omega_list[:-1],plateau,marker='o', linestyle='None')
ax2.plot(Omega_list[:-1],plateau_fit,linestyle='--',label=rf'$\tau(\Omega/J)=exp({k:.2f}\Omega/J+{b:.2f})$')
ax2.text(4,4*1e7,f'$R^2={r_squared:.4f}$',fontsize=10)

ax2.set_xlabel(r'$\Omega/J$')
ax2.set_ylabel(r'$\tau$')
ax2.legend()

fig2.suptitle('Prethermal Plateau Lifetime vs Driving Frequency')




plt.tight_layout()
plt.show()




fig1.savefig('1st.pdf', bbox_inches='tight')
fig2.savefig('2nd.pdf', bbox_inches='tight')
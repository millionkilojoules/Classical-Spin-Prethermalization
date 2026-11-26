import numpy as np
from functions import J,h,g,N,M,l,Omega_list,floquet_step,H_eff,make_initial_state
import time
start_time = time.time()



data=np.load('data/initial_spin.npz')
s_A,c_A,s_B,c_B=data['sols'][0]
theta_A=np.arctan2(s_A,c_A)%(2*np.pi)
theta_B=np.arctan2(s_B,c_B)%(2*np.pi)
print(f'theta_A={theta_A:.5f},theta_B={theta_B:.5f}')
E_cell=(2*J*s_A*s_B+h*(s_A+s_B)+g*(c_A+c_B))/2
E_GS_N=E_cell*N
print(f'E_GS/N={E_cell:.5f}')



#Initial spin
theta_0=np.tile([theta_A,theta_B],N//2)
S_ideal=np.column_stack((np.cos(theta_0),np.zeros_like(theta_0),np.sin(theta_0)))


rng=np.random.default_rng(12345)
initial_states=[make_initial_state(theta_0,rng) for _ in range(M)]

E_GS=H_eff(S_ideal,J,h,g)
print("E_GS =", E_GS)
E_inf=0.0

Q_all=np.zeros((len(Omega_list),l//10+1))
delta_Q_all=np.zeros((len(Omega_list),l//10+1))

H_eff_2_beta=N/4*(1/9*J+1/3*h+1/3*g)

max_norm_error = 0.0



#Q_simulation
for k,Omega in enumerate(Omega_list):
	T=2*np.pi/Omega

	energy_sum=np.zeros(l//10+1)
	energy_sum_2=np.zeros(l//10+1)

	for r in range(M):
		S=initial_states[r].copy()
		energy_sum[0]+=H_eff(S,J,h,g)
		energy_sum_2[0]+=H_eff(S,J,h,g)**2

		for n in range(1,l//10+1):
			S=floquet_step(S,J,h,g,T)
			energy_sum[n]+=H_eff(S,J,h,g)
			energy_sum_2[n]+=H_eff(S,J,h,g)**2

			norm=np.linalg.norm(S,axis=1)
			error=np.max(np.abs(norm-1))

			max_norm_error=max(max_norm_error,error)

	E_mean=energy_sum/M
	E_mean_2=energy_sum_2/M
	variance=E_mean_2-E_mean**2

	Q_all[k,:]=((E_mean-E_GS)/(E_inf-E_GS))
	delta_Q_all[k,:]=np.sqrt((np.maximum(variance,0))/H_eff_2_beta)
	old_data = np.load("data/result.npz")
	S_z = old_data["S_z"]
	old_data.close()
	np.savez('data/result.npz',S_z=S_z,Q_all=Q_all,delta_Q_all=delta_Q_all)
	print("Omega/J =", Omega / J, "finished")

cycles=np.arange(0,l+1,10)
np.savez('data/cycles.npz',cycles=cycles)



end_time = time.time()
print(f'Total time: {end_time-start_time:.2f} seconds')

print("Maximum spin norm error =", max_norm_error)



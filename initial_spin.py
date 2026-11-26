#Solve the initial spin

import numpy as np
from scipy.optimize import root
from functions import J,h,g



def F(v):
	s_A,c_A,s_B,c_B = v
	f1=2*J*c_A*s_B+h*c_A-g*s_A
	f2=2*J*s_A*c_B+h*c_B-g*s_B
	f3=s_A**2+c_A**2-1.0
	f4=s_B**2+c_B**2-1.0

	return np.array([f1,f2,f3,f4],dtype=float)

sols=[]

for a in np.linspace(0,2*np.pi,144,endpoint=False):
	for b in np.linspace(0,2*np.pi,144,endpoint=False):
		v0=np.array([np.sin(a),np.cos(a),np.sin(b),np.cos(b)])
		r=root(F,v0,method='hybr',options={'xtol':1e-12})

		if not r.success:
			continue
		v=r.x
		if np.linalg.norm(F(v))>1e-9:
			continue

		if abs(v[0]**2+v[1]**2-1)>1e-8:
			continue
		if abs(v[2]**2+v[3]**2-1)>1e-8:
			continue

		s_A,c_A,s_B,c_B=v
		H=np.array([
			[-J*s_A*s_B-0.5*h*s_A-0.5*g*c_A,c_A*c_B],
			[c_A*c_B,-J*s_A*s_B-0.5*h*s_B-0.5*g*c_B]
		])


		if not any(np.linalg.norm(v - u) < 1e-6 for u in sols):
			if np.all(np.linalg.eigvalsh(H) > 0):
				sols.append(v.copy())
				break
print(sols)
np.savez('data/initial_spin.npz',sols=sols)
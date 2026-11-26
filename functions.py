import numpy as np



J=1.0
h=0.809
g=0.9045

N=100
M=10
l=int(1e8)

Omega_list=J*np.array([3.8,4.0,4.2,4.4,4.6])


#Floquet simulation
def floquet_step(S,J,h,g,T):

	def tau1(S,J,h,T):
		z = S[:,2]
		kappa=J*(np.roll(z,1)+np.roll(z,-1))+h
		phi=kappa*T/2

		x=S[:,0]
		y=S[:,1]

		S_new=S.copy()
		S_new[:,0]=x*np.cos(phi)-y*np.sin(phi)
		S_new[:,1]=x*np.sin(phi)+y*np.cos(phi)

		return S_new

	def tau2(S,g,T):
		phi=g*T/2

		y=S[:,1]
		z=S[:,2]

		S_new=S.copy()
		S_new[:,1]=y*np.cos(phi)-z*np.sin(phi)
		S_new[:,2]=y*np.sin(phi)+z*np.cos(phi)

		return S_new

	for i in range(10):
		S=tau1(S,J,h,T)
		S=tau2(S,g,T)

	return S

#Effective Hamiltonian simulation
def H_eff(S,J,h,g):

	x=S[:,0]
	z=S[:,2]
	energy=0.5*np.sum(J*z*np.roll(z,-1)+h*z+g*x)

	return energy

#Add small noise to the azimuthal angle of each spin
def make_initial_state(theta_0,rng):
	delta=rng.uniform(-np.pi/100,np.pi/100,size=len(theta_0))
	theta=theta_0+delta

	S=np.column_stack((np.cos(theta),np.zeros(len(theta)),np.sin(theta)))

	return S

#Linear_fitting function
def linear_fitting(t,y):
	k,b=np.polyfit(t,y,1)
	y_fit=k*t+b
	residual=y-y_fit
	r_squared=1-np.sum(residual**2)/np.sum((y-np.mean(y))**2)

	return y_fit,r_squared,k,b
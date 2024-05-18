import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import time as tim
from multiprocessing import Pool


Dimk = 21 #dimension of the hilbert space
iterations=1
beta = 0 #inverse temparature



#defining the goe function
def Generate_GOE(n):
    """Creates nxn GOE"""
    Lambda = np.random.normal(scale=np.sqrt(2/n),size=[n,n])
    G = (Lambda+Lambda.T)/2
    return G



datai = []
for l in range(iterations):
    Hkk = Generate_GOE(Dimk)  #making the hamiltonian 
    #print(Hkk)


    #making krylov subspace
    Kryik = np.zeros(Dimk) 
    Kryik[0] = 1.
    Kryik = expm(- beta * Hkk/2 )@Kryik    #For making a TFD state
    norm = np.sqrt(np.transpose(np.conj(Kryik))@Kryik)
    Kryik= Kryik/norm
    Kryk = []
    for i in range(0, Dimk):
            Kryk.append(np.dot(np.linalg.matrix_power(Hkk, i), Kryik))
    Kryk = np.array(Kryk)
    Kryk = np.linalg.qr(Kryk.T)[0].T 


    #defining the Unitary
    def Ua(t):
            return expm(1j *t * Hkk * np.sqrt(Dimk) )


    #defining the density Matrix
    def rho_eigenkk(t):
            Ua_t = Ua(t)
        
            Kryk_0 = np.transpose([Kryk[0]])
            return (
                np.conj(np.transpose(Ua_t))
                @ Kryk_0
                @ np.conj([Kryk[0]])
                @ Ua_t
            )



    #defining the wigner function
    def wsykenkk(Dimk, t, x, y):
        def Akk(a1, a2):
            result = np.zeros((Dimk, Dimk), dtype=np.complex128)
            for l in range(Dimk):
                for lp in range(Dimk):
                    if (l+lp)%Dimk != (2*a1)%Dimk: continue
                    fact = np.exp(2j * np.pi * (a2 * (l - lp) / Dimk))
                    result += fact * np.transpose(np.conj(np.transpose([Kryk[lp]]))@ [Kryk[l]])
            return result
        return (np.abs(np.trace(Akk(x, y) @ rho_eigenkk(t)))/Dimk)
         



    #parallelization of the code (you don't need to worry)
    if __name__ == '__main__':
        begin = tim.time()
        num_processes = 32 # Adjust as needed (total number of cpu you want )
        timexy = [(Dimk,time,x,y) for time in np.arange(0.0, 4.0, 0.1) for x in range(Dimk) for y in range(Dimk)]
        s = []
        with Pool(num_processes) as pool:
            wsykenkk_steps = pool.starmap(wsykenkk, timexy)
        for time in np.arange(0.0, 4.0, 0.1):
            list = []
            for ii in range(len(timexy)):
                #print(ii)
                if timexy[ii][1] == time: list.append(wsykenkk_steps[ii])
            s.append((time, np.sum(list)))
        datai.append(s)
        end = tim.time()
        print((end - begin)/60)


#making the data
Data = np.sum(datai, axis=0)/iterations

#just set the folder path , where you wann'a save the data
np.save('D:\Project IX\GOE code/data_dim' + str(Dimk) + 'beta='+str(beta)+'.npy', Data)

#plotiing of the graph
plt.plot(Data[:, 0], Data[:, 1])

#just set the folder path , where you wann'a save the figure
plt.savefig('D:\Project IX\GOE code/data_dim' + str(Dimk) + 'beta='+str(beta)+'.pdf')
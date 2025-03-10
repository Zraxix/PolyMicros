import numpy as np
from scipy.spatial import KDTree
import pickle

'''
NEEDS DOCUMENTATION
'''

class ROGSH432():
    def __init__(self, path = "/home/azureuser/Projects/TwoPSGen/GSHTrees/"):
        self.rogsh = np.load(path+"rogsh_432.npy")
        self.euler = np.load(path+"euler_432.npy")

        with open(path+'KDEuler_432.pkl', 'rb') as f:
            self.KDEuler = pickle.load(f)

        with open(path+'KDRogsh_432.pkl', 'rb') as f:
            self.KDRogsh = pickle.load(f)
    
    def CHullProj(self, rogsh):
        _, i = self.KDRogsh.query(rogsh)
        return self.rogsh[i], i

    def Rogsh2Euler(self,rogsh):
        _, i = self.KDRogsh.query(rogsh)
        return self.euler[i] 
    
    @staticmethod
    def ROGSH1(phi1, phi, phi2):
        """
        Method for the Cubic Triclinic GSH function corresponding to the index set: 4	-4	1

        """
        return 1/192*3**(1/2)*((np.cos(phi)-1)**4*np.exp(-4*np.sqrt(-1+0j)*(phi1-phi2))+((1+np.cos(phi))**2*np.exp(-4*np.sqrt(-1+0j)*(phi1+phi2))+14*np.exp(-4*np.sqrt(-1+0j)*phi1)*(np.cos(phi)-1)**2)*(1+np.cos(phi))**2)*2**(1/2)*5**(1/2)

    @staticmethod
    def ROGSH2(phi1, phi, phi2):
        """
        Method for the Cubic Triclinic GSH function corresponding to the index set: 4	0	1

        """
        return 5/48*((np.cos(phi)**4-2*np.cos(phi)**2+1)*np.cos(4*phi2)+7*np.cos(phi)**4-6*np.cos(phi)**2+3/5)*7**(1/2)*3**(1/2)
    
    @staticmethod
    def ROGSH3(phi1, phi, phi2):
        """
        Method for the Cubic Triclinic GSH function corresponding to the index set: 12	0	2

        """
        return 245157/13434880*7**(1/2)*17**(1/2)*23**(1/2)*3**(1/2)*2**(1/2)*13**(1/2)*(1025/735471*(np.cos(phi)-1)**6*(1+np.cos(phi))**6*np.cos(12*phi2)+(np.cos(phi)**8-28/23*np.cos(phi)**6+10/23*np.cos(phi)**4-20/437*np.cos(phi)**2+5/7429)*(1+np.cos(phi))**2*(np.cos(phi)-1)**2*np.cos(4*phi2)+14/969*(1+np.cos(phi))**4*(np.cos(phi)**4-6/23*np.cos(phi)**2+1/161)*(np.cos(phi)-1)**4*np.cos(8*phi2)+182/99*np.cos(phi)**12+14/22287+4550/7429*np.cos(phi)**4-3640/1311*np.cos(phi)**6+130/23*np.cos(phi)**8-364/69*np.cos(phi)**10-364/7429*np.cos(phi)**2)*19**(1/2)*41**(1/2)

    def ROGSH(self, data):
       
        r1 = self.ROGSH1(data[...,0], data[...,1], data[...,2]).real
        r2 = self.ROGSH2(data[...,0], data[...,1], data[...,2]).real
        r3 = self.ROGSH3(data[...,0], data[...,1], data[...,2]).real

        return np.stack([r1,r2,r3],axis=-1)
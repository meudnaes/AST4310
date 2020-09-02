from numpy import exp, loadtxt, empty, nan, copy, newaxis, sum, isnan, where
from astropy.constants import k_B
from astropy.units import Kelvin, cm, spectral
from numpy.ma import fix_invalid
from matplotlib.pyplot import plot, show
import numpy as np

class Atom:
    """
    Reads atomic data, calculates level populations according to Boltzmann's law,
    and ionisation fractions according to Saha's law.
    """
    
    def __init__(self, atomfile = None):
        """
        Parameters
        ----------
        atomfile : string, optional
            Name of file with atomic data. If not present, atomic data needs 
            to be loaded with the .read_atom method.
        """
        self.loaded = False
        if atomfile:
            self.read_atom(atomfile)
        
    def read_atom(self, filename):
        """
        Reads atom structure from text file.
        
        Parameters
        ----------
        filename: string
            Name of file with atomic data.
        """
        tmp = loadtxt(filename, unpack = True)
        self.n_stages = int(tmp[2].max()) + 1
        # Get maximum number of levels in any stage
        self.max_levels = 0
        for i in range(self.n_stages):
            self.max_levels = max(self.max_levels, (tmp[2] == i).sum())
        # Populate level energies and statistical weights
        # Use a square array filled with NaNs for non-existing levels
        chi = empty((self.n_stages, self.max_levels))
        chi.fill(nan)
        self.g = copy(chi)
        for i in range(self.n_stages):
            nlevels = (tmp[2] == i).sum()
            chi[i, :nlevels] = tmp[0][tmp[2] == i]
            self.g[i, :nlevels] = tmp[1][tmp[2] == i]
        # Put units, convert from cm^-1 to Joule
        chi = (chi / cm).to('aJ', equivalencies = spectral())
        # Save ionisation energies, saved as energy of first level in each stage
        self.chi_ion = chi[:, 0].copy()
        # Save level energies relative to ground level in each stage
        self.chi = chi - self.chi_ion[:, newaxis]
        self.loaded = True

    def compute_partition_function(self, temperature):
        if not self.loaded:
            raise ValueError("Missing atom structure, please load atom with read_atim()")
        temp = temperature[np.newaxis, np.newaxis]
        return np.nansum(self.g[..., np.newaxis]*np.exp(-self.chi[..., np.newaxis]/(k_B*temp)),axis=1)    
    
    def cal_partition(self, T):
        """
        Calculates the partition function for the atom using the data in the text file
        and assuming hydrogenic atom and thermodynamic equilibrium

        Parameters
        ----------
        T: float or int
            Temperature of medium around atom
        """
        self.U_r = np.nansum(self.g[..., np.newaxis] * exp(- self.chi[..., np.newaxis] / (k_B * T)), 1)

if __name__ == "__main__":
    #Kalsium = Atom(atomfile = "Ca_atom.txt")
    temp = np.linspace(100, 175000, 10) * Kelvin
    #Kalsium.cal_partition(temp)
    #print(Kalsium.U_r)
    #print("---------------------------------------------------------")
    #print(Kalsium.compute_partition_function(temp))
    print(temp)
    print(temp[:, np.newaxis])
    print(temp[..., np.newaxis])
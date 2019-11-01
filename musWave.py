import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider
import scipy
import scipy.ndimage
import numpy as np

def load(name):
    npzfile=np.load(name)
    return Wave(npzfile['arr_0'],npzfile['arr_1'],npzfile['arr_2'])

class BaseArray(object):
    def __init__(self, array, sampling=None):

        self.array=np.array(array,dtype=complex)

        if len(self.array.shape)!=2 | len(self.array.shape)!=3:
            raise RuntimeError('Only 2d and 3d arrays are allowed')

        if sampling is not None:
            if len(self.array.shape)!=len(sampling):
                raise RuntimeError('Array shape does not match number of sampling entries')

        self.sampling=sampling
        self.offset=(0,0,0)

        self.refs=[]

    def get_dimensions(self):
        dimensions=(self.sampling[0]*self.array.shape[0],self.sampling[1]*self.array.shape[1])
        if len(self.array.shape)==3:
             dimensions+=(self.sampling[2]*self.array.shape[2],)
        return dimensions

    def get_extent(self):
        extent=[self.offset[0],self.array.shape[0]*self.sampling[0]+self.offset[0],
                self.offset[1],self.array.shape[1]*self.sampling[1]+self.offset[1]]
        if len(self.array.shape)==3:
            extent+=[self.offset[2],self.array.shape[2]*self.sampling[2]+self.offset[2]]
        return extent

    def get_reciprocal_extent(self):

        dkx=1/(self.sampling[0]*self.array.shape[0])
        dky=1/(self.sampling[1]*self.array.shape[1])

        extent=[-1/(2*self.sampling[0]),1/(2*self.sampling[0])-dkx,
                -1/(2*self.sampling[1]),1/(2*self.sampling[1])-dky]

        if not self.array.shape[0]%2==0:
            extent[0]-=.5*dkx
            extent[1]-=.5*dkx
        if not self.array.shape[1]%2==0:
            extent[2]-=.5*dky
            extent[3]-=.5*dky
        return extent

class Wave(BaseArray):

    def __init__(self, array, energy, sampling=None, periodic_xy=True):
        BaseArray.__init__(self, array, sampling)
        self.energy = energy
    
    @property
    def shape(self):
        return self.array.shape
    
    @property
    def wavelength(self):
        return 0.38783/np.sqrt(self.energy+9.78476*10**(-4)*self.energy**2)

    def z_slice(self,ind=-1):

        if len(self.array.shape)==2:
            raise RuntimeError('z_slice() only works for 3d wavefunctions')

        return Wavefunction(self.array[:,:,ind],self.energy,self.sampling[:2],self.offset)

    def apply_ctf(self,ctf):
        return ctf.apply(self)

    def resample(self,sampling):

        if len(self.array.shape)==3:
            raise RuntimeError('resample() only works for 2d wavefunctions')

        if not isinstance(sampling, (list, tuple)):
            sampling=(sampling,)*2

        zoom=(self.sampling[0]/sampling[0],self.sampling[1]/sampling[1])

        real = scipy.ndimage.interpolation.zoom(np.real(self.array), zoom)
        imag = scipy.ndimage.interpolation.zoom(np.imag(self.array), zoom)

        sampling=(self.array.shape[0]*self.sampling[0]/real.shape[0],
                    self.array.shape[1]*self.sampling[1]/real.shape[1])

        return Wavefunction(real+1.j*imag,self.energy,sampling,self.offset)

    def save(self,name):
        np.savez(name,self.array,self.energy,self.sampling)
        
    def viewPixel(wave,method='real'):
        if len(wave.array.shape)==3:
            array=np.sum(wave.array,axis=2)
            extent=wave.get_extent()[:4]
        else:
            array=wave.array
            extent=wave.get_extent()

        reciprocal_space=False
        if method == 'amplitude':
            img=np.abs(array)
        elif method == 'real':
            img=np.real(array)
        elif method == 'imaginary':
            img=np.imag(array)
        elif method == 'phase':
            img=np.angle(array)
        elif method == 'intensity':
            img=np.abs(array)**2
        elif method == 'diffraction pattern':
            img=np.log(np.abs(np.fft.fftshift(np.fft.fft2(array)))**2)
            method = method + ' (log scale)'
            reciprocal_space=True
        elif method == 'diffractogram':
            img=np.log(np.abs(np.fft.fftshift(np.fft.fft2(np.abs(array)**2))))
            method += ' (log scale)'
            reciprocal_space=True
        else:
            raise RuntimeError('Unknown method: {0}'.format(method))
        
        print("hey3")
        print(img.size)
        return img
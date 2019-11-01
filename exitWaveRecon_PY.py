#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from pyqstem.imaging import CTF
from reconstruct import reconstruct
from skimage import io
#from pyqstem import wave as PyWave
import musWave


class ExitWaveRecon:
    Name = ''
    day = ''
    setNo = ''
    fName = ''
    path = ''
    # Image Size
    size_nm = 0
    size_pixel = 0
    num_images = 0
    # Camera Param
    energy = 0
    in_Cs = 0
    in_focal_spread = 0
    in_cut_off = 0
    ctfs = []
    sampling = 0
    topDir = ''
    dpi = 0
    def __init__(self):
        print('Time for some exit wave reconstruction')

    def setup(self,_topDir,_name,_day,_setNo):
        self.Name = _name
        self.day = _day
        self.setNo = _setNo
        self.fName = self.Name +'.tif'
        self.topDir = _topDir
        self.path = _topDir + '/' + self.day + '/' + self.setNo + '/'

    def setupParams(self,_size_nm,_size_pixel,_num_images,_energy,_in_Cs,_in_focal_spread,_in_cut_off):
        # Image Size
        self.size_nm = _size_nm
        self.size_pixel = _size_pixel
        self.num_images = _num_images
        # Camera Param
        self.energy = _energy
        self.in_Cs = _in_Cs
        self.in_focal_spread = _in_focal_spread
        self.in_cut_off = _in_cut_off

    def genCTFS(self):
        defocus = np.linspace(-((self.num_images - 1) * 10), ((self.num_images - 1) * 10),self.num_images)
        for i,d in enumerate(defocus):
            ctf = CTF(defocus=d,Cs=self.in_Cs,focal_spread=self.in_focal_spread)
            #,aberrations={"a22": 0,"phi22": 0,"a60": 50000,"a33": 0,"phi33": 0,"a31": 0,"phi31": 0,"a44": 0,"phi44": 0\
            #             ,"a42": 0,"phi42": 0,"a55": 0,"phi55": 0,"a53": 0,"phi53": 0,"a51": 0,"phi51": 0,\
             #             "a66": 0,"phi66": 0,"a64": 0,"phi64": 0,"a62": 0,"phi62": 0,"a60": 0})
            self.ctfs.append(ctf)

    def recon(self):
        images = io.imread(self.path + self.fName)
        self.sampling = float(self.size_nm/self.size_pixel)
        reconstructed = reconstruct(images, self.ctfs, self.energy, self.sampling, tolerance=1e-06,
                                    cutoff=self.in_cut_off, maxiter=400, epsilon=1e-12)
        reconstructed.save(self.path + self.Name)

    def genGraph(self, _myPyWave, _im_type):
        graph = _myPyWave.viewPixel(method=_im_type)
        height, width = graph.shape
        figsize = width / float(self.dpi), height / float(self.dpi)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(graph, cmap='gray', interpolation='nearest')
        ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
        fig.savefig(self.path + '/' + self.Name + '_' + _im_type + '.png', dpi=self.dpi, transparent=True)

    def genFigfromNPZ(self, _dpi):
        self.dpi = _dpi
        full_path = self.path + self.Name + '.npz'
        myPyWave = musWave.load(full_path)
        self.genGraph(myPyWave, 'diffractogram')
        self.genGraph(myPyWave, 'phase')
        self.genGraph(myPyWave, 'amplitude')


if __name__ == '__main__':
    e = ExitWaveRecon()
    e.setup('/Volumes/mus_ML_back/musTEAM/test/images', 'FocalSeries1-AFTERS-180KX', 'day2', 'set4')
    e.setupParams(470.4, 1024, 51, 80, -508, 10, 0.6)
    e.genCTFS()
    e.recon()
    e.genFigfromNPZ(1200)

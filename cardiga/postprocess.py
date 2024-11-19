from nutils import export
import numpy as np

class Graph:

    # Conversion parameters
    __ml   = 1e-6        # [m3]/[ml]
    __mm   = 1e-3        # [m] /[mm]
    __mmHg = 133.3223684 # [Pa]/[mmHg]
    __kPa  = 1e3         # [Pa]/[kPa]
    __ms   = 1e-3        # [ms]/[s] 
    __s    = 1           # [s] /[s]


    def residual(self,resnorm):
        with export.mplfigure('Convergence.png') as fig:            
            ax = fig.add_subplot(111)
            ax.plot(resnorm, 'r-^',label='Actual behavior')
            if len(resnorm) != 0:
                exp = [0,1]+np.cumprod(3*[2]).tolist()
                ax.plot(range(1,len(exp)+1),[10**-i for i in exp ], 'k--',label='Quadratic')
            ax.set_ylabel('Residual norm (absolute)')
            ax.set_xlabel('Newton iterations [-]')
            ax.set_yscale('log')
            ax.grid()
            ax.legend()
        return
    
    def volume_time():
        return
    
    def pressure_time(self, time, P=list(), label=("LV","Art","Ven"), color=("r","b","k")):
        with export.mplfigure('Pressures_leftventricle.png') as fig:            
            ax = fig.add_subplot(111)
            for pres, lab, col in zip(P,label,color):
                ax.plot(time/self.__ms, pres/self.__mmHg, col, label=lab)
            ax.legend()
            ax.set_ylabel('$P$ [mmHg]')
            ax.set_xlabel('$t$ [ms]')
            #ax.set_xlim([0,12])
        return
    
    def volumeflow_time():
        return
    
    # Provide P in [Pa] and V in [m3]
    def pressure_volume(self, V=list(), P=list(), label=("LV","RV"), color=("r","b")):
        assert len(P) == len(V), "Different lengths for pressure and volume input"
        with export.mplfigure('PressureVolume.png') as fig:            
            ax = fig.add_subplot(111)
            for pres, vol, lab, col in zip(P,V,label,color):
                ax.plot(vol/self.__ml, pres/self.__mmHg, col, label=lab)
            ax.set_ylabel('$p$ [mmHg]')
            ax.set_xlabel('$V$ [ml]')
            ax.grid()
            ax.legend()
        return
    

    
    def total_volume(self,time,V):
        with export.mplfigure('TotalVolume_in_system.png') as fig:
            ax = fig.add_subplot(111)
            ax.plot(time/self.__ms, V/self.__ml, 'r-', label='V_{tot}=5 [l]')
            ax.set_ylabel('$V_{tot}$ [ml]')
            ax.set_xlabel('$t$ [ms]')
        return
    

    def overview(self, time, P=list(), Q=list(), V=list(), color=("r","b","k"), plabel=("LV","Art","Ven"), qlabel=("LV","Art","Ven"), vlabel=("LV",), filename='Leftventricle'):
        with export.mplfigure(f'Result_{filename}.png') as fig:            
            ax = fig.add_subplot(311)
            for pres, col, plab in zip(P,color,plabel):
                ax.plot(time/self.__ms, pres/self.__mmHg, col, label=plab)
            ax.legend()
            ax.set_ylabel('$P$ [mmHg]')
            #ax.set_xlim([0,12])
            
            ax = fig.add_subplot(312)
            for vol, col, vlab in zip(V,color,vlabel):
                ax.plot(time/self.__ms, vol/self.__ml, col, label=vlab)
            ax.set_ylabel('$V$ [ml]')
            #ax.set_xlim([0,12])
                        
            
            ax = fig.add_subplot(313)
            for qflow, col, qlab in zip(Q,color,qlabel):
                ax.plot(time/self.__ms, qflow/(self.__ml/self.__s), col, label=qlab)
            ax.legend()
            ax.set_ylabel('$q_{flow}$ [ml/s]')
            ax.set_xlabel('$t$ [ms]')
        return
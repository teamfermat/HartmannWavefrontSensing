# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:40:41 2023

@author: ulm
"""

#OpenSimulation

import pickle
import matplotlib.pyplot as plt
# datetime object containing current date and time
# name = "M:/Arbeitsverzeichnis/OE320/ulm_ab/Python/Simulation21062023 18_40_29"
         
# #to load the container c
nameSaveFiles = "Simulation13072023 13_08_17"
f = open(nameSaveFiles + '.pkl', 'rb')
sim = pickle.load(f)
f.close()

#%%
# def plotWavefrontErrorMap(W,wref,HP):

    #Saving (as template)
#sim={'wavelength':wavelength,'defocus':defocus,'holesPerRow':holesPerRow,\
    #'hd_len':hd_len,'detector_distance':detector_distance,\
    #'sigma':sigma, 'fom':fom,'scaling':scaling,'noiseDivArray':noiseDivArray}

#read out
noiseDivArray = sim['noiseDivArray']
wavelength=sim['wavelength']
defocus=sim['defocus']
# holesPerRow=sim['holesPerRow']
# hd_len=sim['hd_len']
        # for HP_i in range (HP_len):
        #     hole_spacing = (1024/(holesPerRow[HP_i]))/112.5 #1.5
        #     hd=np.linspace(0.1,hole_spacing,hd_len)

detector_distance=sim['detector_distance']
fom=sim['fom']
sigma=sim['sigma']
scaling=sim['scaling']
noiseDivArray=sim['noiseDivArray']
# xShifts=sim['xShifts']
# yShifts=sim['yShifts']

 
noise_len = 10#len(noiseDivArray)
# plotWavefrontErrorMap(W,wref,HP)
#%%
plt.figure(figsize=(7,5))

for i in range (noise_len):
    plt.plot(defocus,fom[0,:,0,0,0,i],label= noiseDivArray[i])
plt.legend()
plt.xscale('log')
plt.xlabel('Zernike coeff [1]')   
plt.ylabel('FOM [%]')   
plt.ylim([60,100])                 
plt.xlim([1e-4,0.1])        
# plt.title('FOM With Noise')

# ax3.lines[0].set_lw(2)
plt.rcParams.update({'font.size': 14})
plt.legend(fontsize="14")
plt.grid("on")
#%%
import numpy as np 
# for j in range (10):
# for j in range (10):
    # figsize=(9,4.8)
plt.figure(figsize=(7,5))

# for i in range (noise_len):
    # np.quantile(a, 0.5, axis=1)
fac = 820
fomHere = fom[0,:,0,0,0,:]
fomCleaned = fomHere * (fomHere>0)
meanThis = np.quantile(fomCleaned,0.5,axis=1)#np.mean(fom[0,:,0,0,0,:],axis = 1)
lower_error = meanThis-np.quantile(fomCleaned,0.25,axis=1)
upper_error = np.quantile(fomCleaned,0.75,axis=1)-meanThis
asymmetric_error = np.zeros([2,20])
asymmetric_error[0,:] = lower_error
asymmetric_error[1,:] = upper_error
#hier weitermachen, fast geschafft


plt.errorbar(defocus*fac,meanThis,yerr=asymmetric_error,color='g',capsize = 2)
plt.legend()
plt.xscale('log')
plt.xlabel('$RMS_{input}$ [$nm$]')     
plt.ylabel('FOM [%]')   
plt.ylim([60,100])              
xLims = np.array([1e-4*0.9,0.04])*fac
plt.xlim(xLims)         
# plt.xlim([1e-4,0.1])        
# plt.title('FOM With Noise')

# ax3.lines[0].set_lw(2)
plt.rcParams.update({'font.size': 16})
plt.legend(fontsize="14")
plt.grid("on")
# from matplotlib.ticker import MaxNLocator
# ax = plt.gca()
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.savefig('Posterfigure3.png', bbox_inches = 'tight')

#%%
plt.figure(figsize=(7,5))

# for i in range (noise_len):
    # np.quantile(a, 0.5, axis=1)
fac = 820
meanThis = np.quantile(fomCleaned,0.5,axis=1)#np.mean(fom[0,:,0,0,0,:],axis = 1)
fomHere = fom[0,:,0,0,0,:]
fomCleaned = fomHere * (fomHere>0)
lower_error = meanThis-np.quantile(fomCleaned,0.25,axis=1)
upper_error = np.quantile(fomCleaned,0.75,axis=1)-meanThis
asymmetric_error = np.zeros([2,20])
asymmetric_error[0,:] = lower_error
asymmetric_error[1,:] = upper_error
#hier weitermachen, fast geschafft


plt.errorbar(defocus*fac,meanThis,yerr=asymmetric_error,color='g',capsize = 2)
plt.legend()
plt.xscale('log')
# plt.xlabel('Zernike coeff [1]')   
plt.ylabel('FOM / %')   
plt.ylim([90,100])       

plt.xlabel('$RMS_{input}$ / nm')   
xLims = np.array([1e-4*0.9,0.04])*fac          
plt.xlim(xLims)        
# plt.title('FOM With Noise')

# ax3.lines[0].set_lw(2)
plt.rcParams.update({'font.size': 16})
plt.legend(fontsize="14")
plt.grid("on")
# from matplotlib.ticker import MaxNLocator
# ax = plt.gca()
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.savefig('Posterfigure3.png', bbox_inches = 'tight')
#%%
fac = 820
plt.figure(figsize=(7,5))
for i in range (noise_len):
    plt.plot(defocus*fac,sigma[0,:,0,0,0,i]*fac,label= noiseDivArray[i])
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$RMS_{input}$ [$nm$]')   
plt.ylabel('$\sigma$ [$nm$]')     
yLims = np.array([10**-5*5,0.04])*fac
plt.ylim(yLims)                   
xLims = np.array([1e-4*0.9,0.04])*fac
plt.xlim(xLims)      
# plt.title('Sigma With Noise')
#fom[a,b,c,d,e], 
#a... wavelength
#b... Zernike coeff
#c... NoA
#d... hole diameter
#%%
plt.figure(figsize=(7,5))
# for i in range (noise_len):
#     plt.plot(defocus,sigma[0,:,0,0,0,i],label= noiseDivArray[i])
fac = 820
sigmaHere = sigma[0,:,0,0,0,:]*fac
meanThis = np.quantile(sigmaHere,0.5,axis=1)#np.mean(fom[0,:,0,0,0,:],axis = 1)
lower_error = meanThis-np.quantile(sigmaHere,0.25,axis=1)
upper_error = np.quantile(sigmaHere,0.75,axis=1)-meanThis
asymmetric_error = np.zeros([2,20])
asymmetric_error[0,:] = lower_error
asymmetric_error[1,:] = upper_error


plt.errorbar(defocus*fac,meanThis,yerr=asymmetric_error,color='g',capsize = 2)
x = np.logspace(-4,-1,2)*fac
y = x
plt.plot(x,y,linestyle = "--",color= 'b', linewidth = 0.5)

y2 = np.ones_like(x)*1.2
plt.plot(x,y2,linestyle = "-.",color= 'k', linewidth = 0.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$RMS_{input}$ / nm')   
plt.ylabel('$\sigma$ / nm')   
yLims = np.array([10**-5*5,0.04])*fac
plt.ylim(yLims)                   
xLims = np.array([1e-4*0.9,0.04])*fac
plt.xlim(xLims)      
plt.grid('on')
# plt.title('Sigma With Noise')
plt.text(0.28,1,"SNR = 1",color= 'b',fontsize = 14,fontweight = 'light')
plt.text(10,1.22,"$\sigma_{Theory}$",color= 'k',fontsize = 14,fontweight = 'light')
#%%
#fom[a,b,c,d,e], 
#a... wavelength
#b... Zernike coeff
#c... NoA
#d... hole diameter
#e... L

# f = plt.gcf()
# fSaveFilesAct = open(nameSaveFiles+"Coeff"+str(defocus[coeff_i])+"Reference.pkl","wb")
# pickle.dump(f,fSaveFilesAct)

# #plot for reconstruction
# interpolate_func=SmoothBivariateSpline(HP.HP_y,HP.HP_x,W)
# z = interpolate_func(xref0,yref0)
# z_hex=get_hex_shape(HP,z,xref,yref,res,holesPerRow[HP_i])
# plot_wavefront(z_hex,'Reconstructed')
# # plot_without_interpolation(HP,W,xref0,yref0,z_hex)
# f = plt.gcf()

exit_pupil_dia=10/2
eps = 0.0001
res=512
xref0=np.linspace(-1-eps,1+eps,res)*exit_pupil_dia
yref0=np.linspace(-1-eps,1+eps,res)*exit_pupil_dia
def plot_wavefrontNm(z,title):
        font = {'family' : 'normal',
        'size'   : 15}
        
        plt.rc('font', **font)     
        plt.figure((4,3))
        plt.pcolor(xref0,yref0,
        z,
        shading='nearest',
        cmap="RdBu_r"
        )
        plt.colorbar(label='[nm]')
        plt.axis('square')
        plt.xlabel('X [mm]')
        plt.ylabel('Y [mm]')
        plt.title(title)
        

for coeff_i in range(0,20,1):
    try:
        with open(nameSaveFiles+"Coeff"+str(defocus[coeff_i])+"Reference.pkl","rb") as fSaveFilesAct:
            w_ref_fitted_hex = pickle.load(fSaveFilesAct)
        plot_wavefrontNm(w_ref_fitted_hex*820,'Reference')
        with open(nameSaveFiles+"Coeff"+str(defocus[coeff_i])+"Reconstructed.pkl","rb") as fSaveFilesAct:
            z_hex = pickle.load(fSaveFilesAct)
        plot_wavefrontNm(z_hex*820,'Reconstructed')
        with open(nameSaveFiles+"Coeff"+str(defocus[coeff_i])+"Errormap.pkl","rb") as fSaveFilesAct:
            wferr_hex = pickle.load(fSaveFilesAct)
        plot_wavefrontNm(wferr_hex*820,'Errormap')
        plt.show()
        break
    except:
        pass
                            
    
    #%%
rmsFamilySS = 2.16205374 #nm
lambdaOverSens = 820/rmsFamilySS
print("$\lambda$ /"+str(lambdaOverSens))

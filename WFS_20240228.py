# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:23:41 2024

@author: Andreas Mathwieser@Fraunhofer IPT
"""

#%%Packages
from itertools import islice

import matplotlib.pyplot as plt
# plt.style.use('seaborn-v0_8-whitegrid')
import numpy as np,math
from scipy.interpolate import SmoothBivariateSpline 
import xlsxwriter
import time
from datetime import datetime
import clr
import os
import winreg
import pickle
from datetime import datetime


#%%Functions
from WFS_Import import *
numHolesArray = np.array([19,
37,\
61,91,\
127,169,\
217,271,\
331,397,\
469,547,\
631,721,\
817,919,\
1027\
])
hdAll = np.array([0.08630631, 0.23261261, 0.2612012 , 0.23219219, 0.20318318,\
       0.183003  , 0.16618619, 0.15441441, 0.14432432, 0.13339339,\
       0.12582583, 0.11825826, 0.11027027, 0.10984985, 0.10228228,\
       0.09471471, 0.0963964 ])
LAll = np.array([100.        , 100.        ,  74.09854927,  56.76488244,\
        45.22561281,  37.10355178,  31.11105553,  26.55477739,\
        22.9889945 ,  20.11655828,  17.78889445,  15.85742871,\
        14.22311156,  12.88594297,  11.69734867,  10.65732866,\
         9.8154077 ])
# for start
for i_holes in range(5,7):
    
# if True:
#     i_holes = 8
    
    #getting parameters for this run
    nrSimulation = i_holes
    num_of_holes = numHolesArray[nrSimulation]
    hd=np.array([hdAll[nrSimulation]])
    detector_distance=np.array([LAll[nrSimulation]])
    
    #add on
    num_of_holesPerRowAll = np.arange(4,38,2)
    holesPerRow = np.array([num_of_holesPerRowAll[nrSimulation]])
    

    
        
    #%% Start zemax
    TicToc = TicTocGenerator() 
    # type_of_connection=input('Please enter the type of connection with OpticStudio (standalone (s) or interactive (i)): ')
    type_of_connection="i"#input('Please enter the type of connection with OpticStudio (standalone (s) or interactive (i)): ')
    print ('Starting opticstudio...')
    if type_of_connection=='standalone' or type_of_connection=='s':
        zos = PythonStandaloneApplication()
        ZOSAPI = zos.ZOSAPI
        TheApplication = zos.TheApplication
        TheSystem = zos.TheSystem
        if not os.path.exists(TheApplication.SamplesDir + "\\API\\Python"):
            os.makedirs(TheApplication.SamplesDir + "\\API\\Python")
        sampleDir = TheApplication.SamplesDir
        testFile = os.path.join(os.sep, sampleDir, r'API\Python\MatlabToPython.zmx')
        TheSystem.New(False)
        TheSystem.SaveAs(testFile)
    elif ((type_of_connection=='interactive') or (type_of_connection=='i')):
        [TheSystem,ZOSAPI]=interactive_connection()
    else:
        raise ValueError ('Please choose either standalone or interactive as a type of connection')
    TheSystemData = TheSystem.SystemData
    TheLDE = TheSystem.LDE
    
    #%%
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y %H_%M_%S")
    nameSaveFiles = "Simulation - third_order" + dt_string
    
    #%% Define variables
    # thisApertureDia = 10
    # factorPro = 1
    # coeff_len=4
    # distortion_type='defocus'
    # defocus= np.logspace(-2,1,coeff_len) 
    # HP_len=1
    
        
    thisApertureDia = 10
    factorPro = 1
    coeff_len=4
    distortion_type='thermal'#'defocus'
    defocus= np.logspace(-2,1,coeff_len) 
    HP_len=1
    
    # num_of_holes=[91]#[1027,721,547,331,127,91,61,37]
    # holesPerRow=[10]#[36,30,26,20,12,10,8,6]
    # num_of_holes=[37]#[1027,721,547,331,127,91,61,37]
    # holesPerRow=[6]#[36,30,26,20,12,10,8,6]
    l_len=1
    hd_len=1 # defined in loop
    w_len=1
    wavelength=[0.82]
    noise_len = 10#2 #10
    noiseDivArray = np.array([0,10,1])#,1,1,1,1,1,1,1,1,1,1])
    
    
    #Simulacao
    print("")
    print("Starting simulation with:" + str(num_of_holes) + " holes")
    print()
    # coeff_len=25
    # defocus= np.logspace(-6,2,coeff_len) 
    coeff_len=5
    defocus= np.logspace(-6,-2,coeff_len) 
    coeff = np.array([0,0,0,0,0,0.0])

    noise_len = 1#2 #10
    noiseDivArray = np.array([100])#,1,1,1,1,1,1,1,1,1,1])
    
    defocus_str = "_".join(["defocus", str(defocus[0]), str(defocus[-1])])
    
    nameSaveFiles = "Simulation - Thermal" + str(numHolesArray[nrSimulation]) + " holes " + defocus_str + " coeff_len " + str(coeff_len)
    
    
    sigma,fom,scaling=np.zeros([w_len,coeff_len,HP_len,hd_len,l_len,noise_len]),np.zeros([w_len,coeff_len,HP_len,hd_len,l_len,noise_len]),np.zeros([w_len,coeff_len,HP_len,hd_len,l_len,noise_len])
    
    #lets see parametes
    res=512
    pixelsize=0.01 #10umx10um
    aperture_value=10*factorPro
    exit_pupil_dia=10/2*factorPro
    d = 10*factorPro #Surface 3 SemiDiameter
    # wavelength=[0.9,1.2,1.5,1.8]
    cases_counter=0
    eps = 0.0001
    xref0=np.linspace(-1-eps,1+eps,res)*exit_pupil_dia
    yref0=np.linspace(-1-eps,1+eps,res)*exit_pupil_dia
    xref,yref = np.meshgrid(xref0,yref0)
    
    
    #%%Get Define simulation tings and surfaces
    # print ('Importing data into opticstudio...')
    TheSystemData.Aperture.ApertureValue = aperture_value
    
    TheSystemData.Advanced.set_ReferenceOPD(0) #Refrence Optical path difference 0=Absolute, 1=Infinity 
    
    TheSystemData.Aperture.set_AFocalImageSpace(True)
    TheAnalyses = TheSystem.Analyses
    TheLDE.InsertNewSurfaceAt(1)
    Surface_1 = TheLDE.GetSurfaceAt(1)
    Surface_2 = TheLDE.GetSurfaceAt(2)
    Surface_1.Comment = 'Zernike Phase to test'
    SurfaceType_Zern = Surface_1.GetSurfaceTypeSettings(ZOSAPI.Editors.LDE.SurfaceType.ZernikeStandardPhase)
    Surface_1.ChangeType(SurfaceType_Zern)
    Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par13).IntegerValue = 56
    Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par14).DoubleValue = thisApertureDia/2
    Surface_2.Comment = 'Hartmann plate'
    Surface_2.SemiDiameter = d
    
    alist=np.zeros(6)
    #%%Extract data from opticstudio
    print('Starting pop analysis')
    newPOP = TheAnalyses.New_Analysis(ZOSAPI.Analysis.AnalysisIDM.PhysicalOpticsPropagation)        
    # pop_analysis(newPOP)
    newWin = TheAnalyses.New_FftMtf()
    
    newPOP.HasAnalysisSpecificSettings
    # newPOP.ModifySettings()
    
    newPOP_Settings = newPOP.GetSettings()
    newPOP_Settings.Wavelength.SetWavelengthNumber(1)
    newPOP_Settings.Wavelength.SetWavelengthNumber(1)
    newPOP_Settings.Field.SetFieldNumber(1)
    newPOP_Settings.StartSurface.SetSurfaceNumber(1)
    newPOP_Settings.EndSurface.SetSurfaceNumber(4)
    newPOP_Settings.SurfaceToBeam = 0.0
    newPOP_Settings.UsePolarization = False
    newPOP_Settings.SeparateXY = False
    newPOP_Settings.UseDiskStorage = False
    newPOP_Settings.SaveOutputBeam = True
    newPOP_Settings.SaveBeamAtAllSurfaces = True
    #newPOP_Settings.OutputBeamFile = fname + "zbf"
    newPOP_Settings.BeamType = ZOSAPI.Analysis.PhysicalOptics.POPBeamTypes.GaussianWaist
    newPOP_Settings.XSampling = ZOSAPI.Analysis.SampleSizes.S_1024x1024
    newPOP_Settings.YSampling = ZOSAPI.Analysis.SampleSizes.S_1024x1024
    newPOP_Settings.DataType = ZOSAPI.Analysis.PhysicalOptics.POPDataTypes.Irradiance
    newPOP_Settings.UseTotalPower = False
    newPOP_Settings.XWidth = 10#1.15 * aper_dia 
    newPOP_Settings.YWidth = 10#1.15 * aper_dia 
    newPOP_Settings.UseTotalPower = False
    newPOP_Settings.SetParameterValue(0, 10)#0.7 * aper_dia)
    newPOP_Settings.SetParameterValue(1, 10)#0.7 * aper_dia)
    print('Starting Wavefront analysis...')    
    new_wf = TheAnalyses.New_Analysis(ZOSAPI.Analysis.AnalysisIDM.WavefrontMap) 
    new_wf.HasAnalysisSpecificSettings
    new_wf_Settings = new_wf.GetSettings()   
    # new_wf_Settings.Surface.SetSurfaceNumber(0)
    new_wf_Settings.set_Sampling(ZOSAPI.Analysis.SampleSizes.S_512x512)
    outputDetailledSteps = 0
    outputPlots = 1 #<<<<<<
    outputFile = 0
    
    for w_i in range (w_len): 
        TheSystemData.Wavelengths.GetWavelength(1).Wavelength = wavelength[w_i]
        for coeff_i in range(coeff_len):
            # if not(coeff_i ==3 or coeff_i==10 or coeff_i == 13):
            #     continue
            if not(distortion_type=='all37'):
                if distortion_type == 'thermal':
                    
                    #higher order wavefronts testing 2,3,4,5
                    coeff[2] = defocus[coeff_i]
                    zc = coeff
                    zero = np.zeros_like(zc)
                else:    
                    zc=defocus[coeff_i]
                    zero=0#[0,0,0,0,0,0,0]
            else:
                zc = coeffAll37
                zero = np.zeros_like(zc)
                
            
            for HP_i in range (HP_len):
                hole_spacing = (1024/(holesPerRow[HP_i]))/112.5 #1.5
                #np.linspace(0.1,hole_spacing,hd_len)
                for hd_i in range (hd_len):
                    aper_dia = (holesPerRow[HP_i]*hole_spacing)+ hd[hd_i]#+1e-2 #for the hartmann plate
                    
                    print('Started case: '+ str(cases_counter+1) +'/' + str(w_len*coeff_len*HP_len*hd_len*l_len*noise_len))
                    cases_counter = cases_counter + 1
                    if outputDetailledSteps: print ('Designing Hartmann Plate...')
                    HP= HartmannPlate(hole_spacing,hd[hd_i],aper_dia)
                    HP.generate_hexagonal_shape(outputPlots) #pass True for plotting the Hartmann plate
                    User_Aper = TheLDE.GetSurfaceAt(2).ApertureData.CreateApertureTypeSettings(ZOSAPI.Editors.LDE.SurfaceApertureTypes.UserAperture)
                    User_Aper.ApertureFile = HP.uda_name
                    Surface_2.ApertureData.ChangeApertureTypeSettings(User_Aper)
                    
                    for l_i in range (l_len):
                        tic(TicToc)
                        L=detector_distance[l_i]
                        Surface_2.Thickness =  L
                        
                        if outputDetailledSteps: print('Getting planewave irradiance...')
                        set_zernike(ZOSAPI, Surface_1,zero,distortion_type) # to get planewave irradiance
                        planewave=extract_data(newPOP)
            
                        if outputDetailledSteps: print('Getting the input wavefront...')
                        User_Aper.ApertureFile = 'None'
                        Surface_2.ApertureData.ChangeApertureTypeSettings(User_Aper)
                        # if zc<=0.01:
                        #     set_zernike(-1*zc,distortion_type)
                        # else:
                        set_zernike(ZOSAPI,Surface_1,zc,distortion_type)
    
                        wavefront_from_os=extract_data(new_wf)
                      
                        if outputDetailledSteps: print('Getting aberrated irradiance...')
                        set_zernike(ZOSAPI,Surface_1,zc,distortion_type)
                        User_Aper.ApertureFile = HP.uda_name
                        Surface_2.ApertureData.ChangeApertureTypeSettings(User_Aper)
                        aberrated=extract_data(newPOP)
                        if outputPlots: plot_diffraction(exit_pupil_dia,planewave, "planewave")
                        if outputPlots: plot_diffraction(exit_pupil_dia,aberrated, "aberrated")
    #%% Post-processing the signal data
                        if outputDetailledSteps: print('Post-processing the signal data...')
        
                        
                        for noise_i in range(noise_len):
                            noiseDivAct = noiseDivArray[noise_i]
                            hp_img = hole_spacing*102.5#(1024/mm[HP_i])
        
                            #without noise
                            if noiseDivAct == 0:
                                [xShifts,yShifts]=getXandYshifts(HP,planewave,aberrated,hp_img) #Get centroid shifts    
                            else:
                                #with noise
                                aberrated_with_noise=poisson(aberrated,noiseDivAct)
                                [xShifts,yShifts]=getXandYshifts(HP,planewave,aberrated_with_noise,hp_img) #Get centroid shifts             
                            
                            if outputDetailledSteps: print("Reconstruction")
                            
                            #reconstructed wavefront
                            W = get_W(HP,xShifts,yShifts,L,pixelsize) #Get the W according to the zonal algorithm
                            middleW = int(len(W)/2)
                            if W[middleW]<W[middleW+1]:
                                W = -W
                            W-=np.min(W)  
                            
                            wref=get_discrete_points(xref,yref,wavefront_from_os,HP,res)
                            middleW = int(len(wref)/2)
                            if wref[middleW]<wref[middleW+1]:
                                wref = -wref
                            wref-=np.min(wref)  
                            
                            #if outputDetailledSteps: print(np.max(wref)/np.max(W))
                            scaling[w_i,coeff_i,HP_i,hd_i,l_i,noise_i]=np.max(wref)/np.max(W)
                            # print("Scaling: " + str(scaling[w_i,coeff_i,HP_i,hd_i,l_i,noise_i]))
                            W=W/wavelength[w_i]*1000 #scaling[w_i,coeff_i,HP_i,hd_i,l_i,noise_i]
                            # W*= scaling[w_i,coeff_i,HP_i,hd_i,l_i,noise_i]
                            # if outputDetailledSteps: print('NoH= '+str(HP.num_of_holes)+ ' , hole diameter= '+str(HP.hs))
                            # if outputDetailledSteps: print('HP diameter= '+str(HP.AperDia))
                            # if outputDetailledSteps: print('Distance to the camera= '+str(detector_distance[l_i]))
                            # if outputDetailledSteps: print('Aberration type: '+distortion_type +' , Zc= '+ str(zc))
                            if outputDetailledSteps:
                                print(f'NoH= {HP.num_of_holes}, hole diameter= {HP.hs}')
                                print(f'HP diameter= {HP.AperDia}')
                                print(f'Distance to the camera= {detector_distance[l_i]}')
                                print(f'Aberration type: {distortion_type}, Zc= {zc}')
                                print(np.max(wref)/np.max(W))
                            
                            sigma[w_i,coeff_i,HP_i,hd_i,l_i,noise_i],fom[w_i,coeff_i,HP_i,hd_i,l_i,noise_i]=get_fom(W,wref,HP)
            
            
                            if noise_i == 0:
                                #plotting start here
                                #plot for input
                                # #for wavefront plot but not consistent with the w-scaling
                                # wf_hex=get_hex_shape(HP,wavefront_from_os,xref,yref,res,holesPerRow[HP_i])
                                # plot_without_interpolation(HP,wref,xref0,yref0,wf_hex)
                                # alternative wavefront plot:
                                w_ref_fitfunc=SmoothBivariateSpline(HP.HP_y,HP.HP_x,wref)
                                w_ref_fitted = w_ref_fitfunc(xref0,yref0)
                                w_ref_fitted_hex = get_hex_shape(HP,w_ref_fitted,xref,yref,res,holesPerRow[HP_i])
                                if outputPlots: plot_wavefront(xref0,yref0,w_ref_fitted_hex,'Reference')
                                # f = plt.gcf()
                                
                                if outputFile: 
                                    fSaveFilesAct = open(nameSaveFiles+"Coeff"+str(defocus[coeff_i])+"Referesnce.pkl","wb")
                                    pickle.dump(w_ref_fitted_hex,fSaveFilesAct)
                                    fSaveFilesAct.close()
                                
                                #plot for reconstruction
                                interpolate_func=SmoothBivariateSpline(HP.HP_y,HP.HP_x,W)
                                z = interpolate_func(xref0,yref0)
                                z_hex=get_hex_shape(HP,z,xref,yref,res,holesPerRow[HP_i])
                                if outputPlots: plot_wavefront(xref0,yref0,z_hex,'Reconstructed '+ str(defocus[coeff_i]))
                                # plot_without_interpolation(HP,W,xref0,yref0,z_hex)
                                # f = plt.gcf()
                                if outputFile: 
                                    fSaveFilesAct = open(nameSaveFiles+'Coeff'+str(defocus[coeff_i])+"Reconstructed.pkl","wb")
                                    pickle.dump(z_hex,fSaveFilesAct)
                                    fSaveFilesAct.close()
                                
                                if outputDetailledSteps: print("W minmax: " + str(np.min(W)) + " " + str(np.max(W)))
                
                                # plot_without_interpolation(HP,W)
                                # plot_without_interpolation(HP,wref)
                                
                                #Plot wavefront error map
                                wferr = np.zeros_like(W)
                                mCB = myColorBar(np.max(wferr)-np.min(wferr))
                                for i in range(len(W)):
                                    wferr[i] = W[i]-wref[i]
                                
                                interpolate_func=SmoothBivariateSpline(HP.HP_y,HP.HP_x,wferr)
                                z = interpolate_func(xref0,yref0)
                                wferr_hex=get_hex_shape(HP,z,xref,yref,res,holesPerRow[HP_i])
                                if outputPlots:plot_wavefront(xref0,yref0,wferr_hex,'Errormap')
                                # f = plt.gcf()
                                if outputFile: 
                                    fSaveFilesAct = open(nameSaveFiles+'Coeff'+str(defocus[coeff_i])+"Errormap.pkl","wb")
                                    pickle.dump(wferr_hex,fSaveFilesAct)
                                    fSaveFilesAct.close()
    
    
                            # #Output time
                            # print('Case '+ str(cases_counter+1) +' out of ' + str(w_len*coeff_len*HP_len*hd_len*l_len*noise_len)+' has been tested completely')
                            # full_time=(30/60)*coeff_len*HP_len*hd_len*l_len*noise_len
                            # toc(TicToc,True)
                            
                            # time_taken=30
                            # time_remaining=round(full_time-((time_taken/60)*cases_counter))/60
                            # if time_remaining < 1:
                            #     print('Time remaining: '+str(round(time_remaining*60))+' minutes')
                            # else:
                            #     print('Time remaining: '+str(round(time_remaining*10)/10)+' hours')
                            # cases_counter+=1
                            
                            # # plotWavefrontErrorMap(W,wref,HP)
                            # #%%
                            # # for j in range (10):
                            plt.figure()
                            
                            # for i in range (noise_len):
                            plt.plot(defocus,fom[0,:,0,0,0,0])#,label= noiseDivArray[i])
                            # plt.legend()
                            plt.xscale('log')
                            plt.xlabel('Zernike coeff [1]')   
                            plt.ylabel('FOM [%]')   
                            plt.ylim([50,100])                 
                            # plt.xlim([1e-3,0.1])        
                            plt.title('FOM With Noise')
                            plt.show()
                            
                            plt.figure()
                            
                            # for i in range (noise_len):
                            plt.plot(defocus,sigma[0,:,0,0,0,0])#,label= noiseDivArray[i])
                            # plt.legend()
                            plt.xscale('log')
                            plt.xlabel('Zernike coeff [1]')   
                            plt.ylabel('Sigma [waves]')   
                            # plt.ylim([50,100])                 
                            # plt.xlim([1e-3,0.1])        
                            plt.title('Sigma With Noise')
                            plt.show()
                            # #%%
                            # plt.figure()
                            # for i in range (noise_len):
                            #     plt.plot(defocus,sigma[0,:,0,0,0,i],label= noiseDivArray[i])
                            # plt.legend()
                            # plt.xscale('log')
                            # plt.yscale('log')
                            # plt.xlabel('Zernike coeff [1]')   
                            # plt.ylabel('Sigma [waves]')   
                            # # plt.ylim([90,100])                 
                            # #plt.xlim([1e-3,0.1])        
                            # plt.title('Sigma With Noise')
                            # #fom[a,b,c,d,e], 
                            # #a... wavelength
                            # #b... Zernike coeff
                            # #c... NoA
                            # #d... hole diameter
                            # #e... L
                            # #%%
                            # # plt.figure()
                            
                            # # for i in range (noise_len):
                            # #     plt.plot(defocus,fom[0,:,1,0,0,i],label= noiseDivArray[i])
                            # # plt.legend()
                            # # plt.xscale('log')
                            # # plt.xlabel('Zernike coeff [1]')   
                            # # plt.ylabel('FOM [%]')   
                            # # plt.ylim([90,100])                 
                            # # # plt.xlim([1e-3,0.1])        
                            # # plt.title('FOM With Noise')
                            # # #%%
                            # # plt.figure()
                            # # for i in range (noise_len):
                            # #     plt.plot(defocus,sigma[0,:,1,0,0,i],label= noiseDivArray[i])
                            # # plt.legend()
                            # # plt.xscale('log')
                            # # plt.yscale('log')
                            # # plt.xlabel('Zernike coeff [1]')   
                            # # plt.ylabel('Sigma [waves]')   
                            # # # plt.ylim([90,100])                 
                            # # #plt.xlim([1e-3,0.1])        
                            # # plt.title('Sigma With Noise')
    
    
    #to save variables in a container
    sim={'wavelength':wavelength,'defocus':defocus,'holesPerRow':holesPerRow,'hd_len':hd_len,'detector_distance':detector_distance,\
          'sigma':sigma, 'fom':fom,'scaling':scaling,'noiseDivArray':noiseDivArray}
    
    # fSaveFiles = open(nameSaveFiles+".pkl","wb")
    # pickle.dump(sim,fSaveFiles)
    # fSaveFiles.close()
    
    if os.path.exists(nameSaveFiles + ".pkl"):
        # Counter for number of files with the same name
        count = 1
        
        # Loop para encontrar um nome único para o arquivo
        while os.path.exists(nameSaveFiles + f"({count}).pkl"):
            count += 1
        
        # Loop to find a unique name for the file
        nameSaveFiles += f"({count})"
    
    fSaveFiles = open(nameSaveFiles + ".pkl", "wb")
    pickle.dump(sim, fSaveFiles)
    fSaveFiles.close()
    # pass #for end
                        # #to load the container c
                        # f = open('c.pkl', 'rb')
                        # c = pickle.load(f)
                        # f.close()

#%%
# def plotWavefrontErrorMap(W,wref,HP):




# plotWavefrontErrorMap(W,wref,HP)
#%%
# for j in range (10):
    # figsize=(9,4.8)
# plt.figure(figsize=(7,3))

# for i in range (noise_len):
#     plt.plot(defocus,fom[0,:,0,0,0,i],label= noiseDivArray[i])
# plt.legend()
# plt.xscale('log')
# plt.xlabel('Zernike coeff [1]')   
# plt.ylabel('FOM [%]')   
# plt.ylim([60,100])                 
# plt.xlim([1e-4,0.1])        
# plt.title('FOM With Noise')

# ax3.lines[0].set_lw(2)
# plt.rcParams.update({'font.size': 16})
# plt.legend(fontsize="14")
# plt.grid("on")
# plt.errorbar(xVec[2:],mean2[2:],std2[2:],color='r',linestyle='--',capsize = 2,label='horizontal')
# plt.errorbar(xVec[2:],mean4[2:],std4[2:],color='g',capsize = 2,label='vertical')
# from matplotlib.ticker import MaxNLocator
# ax = plt.gca()
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.savefig('Posterfigure3.png', bbox_inches = 'tight')

#%%
# for j in range (10):
    # figsize=(9,4.8)
# plt.figure(figsize=(7,3))

# # for i in range (noise_len):
# plt.errorbar(defocus,np.mean(fom[0,:,0,0,0,:]),np.std(fom[0,:,0,0,0,:]))#,color='g',capsize = 2)#,label= noiseDivArray[i])
# # plt.legend()
# plt.xscale('log')
# plt.xlabel('Zernike coeff [1]')   
# plt.ylabel('FOM [%]')   
# plt.ylim([60,100])                 
# plt.xlim([1e-4,0.1])        
# plt.title('FOM With Noise')

# # ax3.lines[0].set_lw(2)
# plt.rcParams.update({'font.size': 16})
# plt.legend(fontsize="14")
# plt.grid("on")
# # from matplotlib.ticker import MaxNLocator
# # ax = plt.gca()
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.savefig('Posterfigure3.png', bbox_inches = 'tight')
# #%%
# plt.figure()
# for i in range (noise_len):
#     plt.plot(defocus,sigma[0,:,0,0,0,i],label= noiseDivArray[i])
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Zernike coeff [1]')   
# plt.ylabel('Sigma [waves]')   
# # plt.ylim([90,100])                 
# #plt.xlim([1e-3,0.1])        
# plt.title('Sigma With Noise')
# #fom[a,b,c,d,e], 
# #a... wavelength
# #b... Zernike coeff
#c... NoA
#d... hole diameter
#e... L
#%%
# plt.figure()

# for i in range (noise_len):
#     plt.plot(defocus,fom[0,:,1,0,0,i],label= noiseDivArray[i])
# plt.legend()
# plt.xscale('log')
# plt.xlabel('Zernike coeff [1]')   
# plt.ylabel('FOM [%]')   
# plt.ylim([90,100])                 
# # plt.xlim([1e-3,0.1])        
# plt.title('FOM With Noise')
# #%%
# plt.figure()
# for i in range (noise_len):
#     plt.plot(defocus,sigma[0,:,1,0,0,i],label= noiseDivArray[i])
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Zernike coeff [1]')   
# plt.ylabel('Sigma [waves]')   
# # plt.ylim([90,100])                 
# #plt.xlim([1e-3,0.1])        
# plt.title('Sigma With Noise')
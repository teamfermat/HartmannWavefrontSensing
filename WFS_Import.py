# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:42:19 2024

@author: Andreas Mathwieser@Fraunhofer IPT
"""

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
total_time = 0

#%%Classes 

def timer(func):
    def wrapper(*args, **kwargs):
        
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        global total_time
        total_time += end - start
        if total_time != 0:
            print(f"Tempo total na função {func.__name__}: {total_time:.2f} segundos")
            total_time = 0
        return result
        
    return wrapper

class PythonStandaloneApplication(object):
    class LicenseException(Exception):
        pass
    class ConnectionException(Exception):
        pass
    class InitializationException(Exception):
        pass
    class SystemNotPresentException(Exception):
        pass
    
    def __init__(self, path=None):
        # determine location of ZOSAPI_NetHelper.dll & add as reference
        aKey = winreg.OpenKey(winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER), r"Software\Zemax", 0, winreg.KEY_READ)
        zemaxData = winreg.QueryValueEx(aKey, 'ZemaxRoot')
        NetHelper = os.path.join(os.sep, zemaxData[0], r'ZOS-API\Libraries\ZOSAPI_NetHelper.dll')
        winreg.CloseKey(aKey)
        clr.AddReference(NetHelper)
        import ZOSAPI_NetHelper
        
        # Find the installed version of OpticStudio
        if path is None:
            isInitialized = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize()
        else:
            # Note -- uncomment the following line to use a custom initialization path
            isInitialized = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(path)
        
        # determine the ZOS root directory
        if isInitialized:
            dir = ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory()
        else:
            raise PythonStandaloneApplication.InitializationException("Unable to locate Zemax OpticStudio.  Try using a hard-coded path.")

        # add ZOS-API referencecs
        clr.AddReference(os.path.join(os.sep, dir, "ZOSAPI.dll"))
        clr.AddReference(os.path.join(os.sep, dir, "ZOSAPI_Interfaces.dll"))
        import ZOSAPI

        # create a reference to the API namespace
        self.ZOSAPI = ZOSAPI

        # create a reference to the API namespace
        self.ZOSAPI = ZOSAPI

        # Create the initial connection class
        self.TheConnection = ZOSAPI.ZOSAPI_Connection()

        if self.TheConnection is None:
            raise PythonStandaloneApplication.ConnectionException("Unable to initialize .NET connection to ZOSAPI")

        self.TheApplication = self.TheConnection.CreateNewApplication()
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException("Unable to acquire ZOSAPI application")

        if self.TheApplication.IsValidLicenseForAPI == False:
            raise PythonStandaloneApplication.LicenseException("License is not valid for ZOSAPI use")

        self.TheSystem = self.TheApplication.PrimarySystem
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")
         
    def __del__(self):
        if self.TheApplication is not None:
            self.TheApplication.CloseApplication()
            self.TheApplication = None
        
        self.TheConnection = None
    
    def OpenFile(self, filepath, saveIfNeeded):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.LoadFile(filepath, saveIfNeeded)
   
    def CloseFile(self, save):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.Close(save)

    def SamplesDir(self):
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException("Unable to acquire ZOSAPI application")

        return self.TheApplication.SamplesDir

    def ExampleConstants(self):
        if self.TheApplication.LicenseStatus == self.ZOSAPI.LicenseStatusType.PremiumEdition:
            return "Premium"
        elif self.TheApplication.LicenseStatus == self.ZOSAPI.LicenseStatusTypeProfessionalEdition:
            return "Professional"
        elif self.TheApplication.LicenseStatus == self.ZOSAPI.LicenseStatusTypeStandardEdition:
            return "Standard"
        else:
            return "Invalid"
    
    def reshape(self, data, x, y, transpose = False):
        """Converts a System.Double[,] to a 2D list for plotting or post processing
        
        Parameters
        ----------
        data      : System.Double[,] data directly from ZOS-API 
        x         : x width of new 2D list [use var.GetLength(0) for dimension]
        y         : y width of new 2D list [use var.GetLength(1) for dimension]
        transpose : transposes data; needed for some multi-dimensional line series data
        
        Returns
        -------
        res       : 2D list; can be directly used with Matplotlib or converted to
                    a numpy array using numpy.asarray(res)
        """
        if type(data) is not list:
            data = list(data)
        var_lst = [y] * x;
        it = iter(data)
        res = [list(islice(it, i)) for i in var_lst]
        if transpose:
            return self.transpose(res);
        return res
    
    def transpose(self, data):
        """Transposes a 2D list (Python3.x or greater).  
        
        Useful for converting mutli-dimensional line series (i.e. FFT PSF)
        
        Parameters
        ----------
        data      : Python native list (if using System.Data[,] object reshape first)    
        
        Returns
        -------
        res       : transposed 2D list
        """
        if type(data) is not list:
            data = list(data)
        return list(map(list, zip(*data)))
   
def interactive_connection():
    #determine the Zemax working directory
    aKey = winreg.OpenKey(winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER), r"Software\Zemax", 0, winreg.KEY_READ)
    zemaxData = winreg.QueryValueEx(aKey, 'ZemaxRoot')
    NetHelper = os.path.join(os.sep, zemaxData[0], r'ZOS-API\Libraries\ZOSAPI_NetHelper.dll')
    winreg.CloseKey(aKey)
    
    # add the NetHelper DLL for locating the OpticStudio install folder
    clr.AddReference(NetHelper)
    import ZOSAPI_NetHelper
    
    pathToInstall = ''
    
    # connect to OpticStudio
    success = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(pathToInstall);
    
    zemaxDir = ''
    if success:
        zemaxDir = ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory();
        print('Found OpticStudio at:   %s' + zemaxDir);
    else:
        raise Exception('Cannot find OpticStudio')
    
    # load the ZOS-API assemblies
    clr.AddReference(os.path.join(os.sep, zemaxDir, r'ZOSAPI.dll'))
    clr.AddReference(os.path.join(os.sep, zemaxDir, r'ZOSAPI_Interfaces.dll'))
    import ZOSAPI
    
    TheConnection = ZOSAPI.ZOSAPI_Connection()
    if TheConnection is None:
        raise Exception("Unable to intialize NET connection to ZOSAPI")
    
    TheApplication = TheConnection.ConnectAsExtension(0)
    if TheApplication is None:
        raise Exception("Unable to acquire ZOSAPI application")
    
    if TheApplication.IsValidLicenseForAPI == False:
        raise Exception("License is not valid for ZOSAPI use.  Make sure you have enabled 'Programming > Interactive Extension' from the OpticStudio GUI.")
    
    TheSystem = TheApplication.PrimarySystem
    if TheSystem is None:
        raise Exception("Unable to acquire Primary system")
    
    def reshape(data, x, y, transpose = False):
        """Converts a System.Double[,] to a 2D list for plotting or post processing
        
        Parameters
        ----------
        data      : System.Double[,] data directly from ZOS-API 
        x         : x width of new 2D list [use var.GetLength(0) for dimension]
        y         : y width of new 2D list [use var.GetLength(1) for dimension]
        transpose : transposes data; needed for some multi-dimensional line series data
        
        Returns
        -------
        res       : 2D list; can be directly used with Matplotlib or converted to
                    a numpy array using numpy.asarray(res)
        """
        if type(data) is not list:
            data = list(data)
        var_lst = [y] * x;
        it = iter(data)
        res = [list(islice(it, i)) for i in var_lst]
        if transpose:
            return transpose(res);
        return res
      
    def transpose(data):
        """Transposes a 2D list (Python3.x or greater).  
        
        Useful for converting mutli-dimensional line series (i.e. FFT PSF)
        
        Parameters
        ----------
        data      : Python native list (if using System.Data[,] object reshape first)    
        
        Returns
        -------
        res       : transposed 2D list
        """
        if type(data) is not list:
            data = list(data)
        return list(map(list, zip(*data)))
    
    print('Connected to OpticStudio')
    
    # The connection should now be ready to use.  For example:
    print('Serial #: ', TheApplication.SerialCode)
    
    # Insert Code Here
    
    # creates a new API directory
    if not os.path.exists(TheApplication.SamplesDir + "\\API\\Python"):
        os.makedirs(TheApplication.SamplesDir + "\\API\\Python")
    
    # Set up primary optical system
    sampleDir = TheApplication.SamplesDir
    
    #! [e01s01_py]
    # Make new file
    testFile = os.path.join(os.sep, sampleDir, r'API\Python\MatlabToPython.zmx')
    TheSystem.New(False)
    TheSystem.SaveAs(testFile)
    #! [e01s01_py]
    
    return(TheSystem,ZOSAPI)

class HartmannPlate():
    
    def __init__(self,hp,hs,AperDia):
        self.hp = hp
        self.hs = hs
        self.AperDia=AperDia
    #@timer
    def generate_hexagonal_shape(self,plot_flag):
    
        aper_dia=self.AperDia#(8*self.hp)+self.hs #for the hartmann plate
        points = []
        
        vertSpacing = self.hp * math.cos(30/180 * math.pi)
        holesPerRow = math.floor(aper_dia/self.hp) #how many holes there are in a row (at max)
        holesPerRow = holesPerRow + ((holesPerRow+1)% 2) # make sure number of cols is odd
        numOfRows = holesPerRow
        minholesPerRow=int((numOfRows+1)/2)
        row_lengths1=np.linspace(minholesPerRow,holesPerRow,minholesPerRow)
        row_lengths2=np.linspace(holesPerRow-1,minholesPerRow,minholesPerRow-1)
        row_lengths=np.concatenate((row_lengths1,row_lengths2))

        # Generate points for each row
        y=-1*((numOfRows-1)/2)*vertSpacing

        start=(minholesPerRow-1)/2
        stop=(holesPerRow-1)/2
        offset1=np.arange(-start,-stop,-0.5)
        offset2=np.arange(-stop,-start+0.5,0.5)
        offset=np.concatenate((offset1,offset2))
        for row in range(numOfRows):
            row_length = row_lengths[row]
            x=0
            for i in range(int(row_length)):
                points.append((x+offset[row]*self.hp, y))
                x = x+self.hp#(i + offset) * (diameter + spacing)
            y = y+vertSpacing
    
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        
        if plot_flag:
            # Scatter plot
            plt.figure()
            plt.scatter(x_coords, y_coords, s=self.hs*100)  # 's' specifies the size of the markers
            
            # Set aspect ratio to equal and add labels
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Hartmann Plate')
            plt.show()
        
        #write the uda file 
        # path = '\Zemax\Objects\Apertures\neu\HartmannPlate.uda'
        filename= 'HartmannPlateAutomation.UDA' #'HP_' + datetime.now().strftime("D_%d_%m_%Y_T_%H_%M_%S") + '.UDA'
        # path = r'C:\Users\ulm-ab\Documents\Zemax\Objects\Apertures\\'  
        # filename = r'M:\Arbeitsverzeichnis\OE320\ulm_ab\Python\HartmannPlate.uda'
        path = r'C:\Users\ulm-gs\Documents\Zemax\Objects\Apertures' 
        path = path + "\\" + filename
        
        
        fid = open(path, 'w+');
        for id in range (len(x_coords)):
            # 13 digits precision
            #prec = 13;
            line = "CIR " + str(x_coords[id])+ " " + str(y_coords[id]) + " " + str(self.hs/2)
            fid.write(line + '\n' )
        fid.close()
        # path = r'C:\Users\ulm\Documents\Zemax\Objects\Apertures\HP_D_20_06_2023_T_13_27_40.uda'
        self.uda_name=filename
        self.HP_x=x_coords
        self.HP_y=y_coords
        self.num_of_holes=len(x_coords)
        
        
        
class Hole():
    
    def __init__(self,hp,a,b):
        self.hp = hp
        self.a = a
        self.b = b    
    
    def defineArea(self, imageTotal,size):
        self.imageSize = size
        self.imageArea = np.zeros((size,size))
        for i in range(size):
            for j in range(size):
                yindex=int(512+i+math.floor(self.a)-size/2)
                xindex=int(512+j+math.floor(self.b)-size/2)
                try:
                    self.imageArea[j,i]=imageTotal[yindex,xindex] 
                except:
                    self.imageArea[j,i]=0
    #@timer
    def plotArea(self):
        plt.imshow(self.imageArea)
        plt.show()
    #@timer           
    def calculateCenter(self):      
        sumx=0
        sumy=0
        sum2=0
        x = np.arange(self.imageSize)-int(self.imageSize/2)
        y = x
        [xx,yy] = np.meshgrid(x,y)
        circle= (np.abs(xx**2+yy**2)<=((self.imageSize**2)/4))
        # plt.figure()
        # plt.imshow(self.imageArea*circle) 
        jR = 0
        for i in range (self.imageSize):
            jR = 0
            for j in range (self.imageSize):
                sumx=sumx+ i*self.imageArea[i,j]**2
                sumy=sumy+ j*self.imageArea[i,j]**2
                sum2=sum2+ (self.imageArea[i,j]**2)*circle[i,j]
                # if circle[i,j] and jR == 0:
                #     jR = 1
                #     plt.plot(i,j, marker='o', color="blue")
        self.xC, self.yC = sumx/sum2 , sumy/sum2
        # plt.figure()
        # plt.plot(self.yC,self.xC, marker='o', color="red")
        # plt.axis('off')
        # plt.show()
       
    def getCenters(self):
        yGlobal = int(512+self.yC + self.a-self.imageSize/2)
        xGlobal = int(512+self.xC + self.b-self.imageSize/2)
        
        return xGlobal, yGlobal
    
    def getImageCenter(self):
        y = int(512+ self.a)
        x = int(512+self.b)
        return x, y    
    
    def getShifts(self):
        return self.xC, self.yC



def get_hex_shape(HP,wavefront,xref,yref,res,holesPerRow):
    wf_hex=np.zeros([res,res])
    s=(holesPerRow/2)*HP.hp 
    hex_area= abs(3*np.sqrt(3)*(s**2)/2)  # or just use: Reconstruction.hexagon_area(HP) 
    for i in range(res):
        for j in range(res):
            if (isInside(xref[i,j], yref[i,j],HP,hex_area,holesPerRow)):
                wf_hex[i,j]=wavefront[i,j]
    return wf_hex

def area( x1,  y1,  x2,  y2,  x3,  y3):
    # Hexagonal Shape using area of hexagon
    return abs((x1*(y2-y3) + x2*(y3-y1)+ x3*(y1-y2))/2)

def isInside(x,y,HP,hex_area,holesPerRow):
    ed2=int(0+holesPerRow/2)
    ed3=int(((HP.num_of_holes-holesPerRow-1)/2)+(holesPerRow))
    ed4=int(HP.num_of_holes-1)
    ed5=int(HP.num_of_holes-1-(holesPerRow/2))
    ed6=int(((HP.num_of_holes-holesPerRow-1)/2))
     
    edges=[0,ed2,ed3,ed4,ed5,ed6,0]
    # print(edges)
    A=0

    for i in range (6):
        x2,y2=HP.HP_x[edges[i]],HP.HP_y[edges[i]]
        x3,y3=HP.HP_x[edges[i+1]],HP.HP_y[edges[i+1]]
        A+= area(x, y, x2, y2, x3, y3)

    return np.abs(A - hex_area) < 0.01

def getAandB(hp_img,HP):
    aper_dia=HP.AperDia
    a=np.zeros(HP.num_of_holes)
    b=np.zeros(HP.num_of_holes)
    # ver_dis=np.sqrt(3)*(hp/2) # distance between two holes vertically
    ver_dis = hp_img * math.cos(30/180 * math.pi)
    
    holesPerRow = math.floor(aper_dia/HP.hp) #how many holes there are in a row (at max)
    holesPerRow = holesPerRow + ((holesPerRow+1)% 2) # make sure number of cols is odd
    numOfRows = holesPerRow
    minholesPerRow=int((numOfRows+1)/2)
    start=(minholesPerRow-1)/2
    stop=(holesPerRow-1)/2
    rows1=np.arange(-start,-stop,-0.5)
    rows2=np.arange(-stop,start+0.5,0.5)
    rows=np.concatenate((rows1,rows2))
    row_lengths1=np.linspace(minholesPerRow,holesPerRow,minholesPerRow)
    row_lengths2=np.linspace(holesPerRow-1,minholesPerRow,minholesPerRow-1)
    row_lengths=np.concatenate((row_lengths1,row_lengths2))    
    holes_sum=0
    row_num=-(holesPerRow-1)/2
    for k in range(HP.num_of_holes):
        holes_sum=0
        row_num=-(holesPerRow-1)/2
        for i in range (numOfRows):
            holes_sum=holes_sum+row_lengths[i]
            if k < holes_sum:
                a[k] = (row_num)*ver_dis
                b[k]= hp_img*rows[i]
                rows[i]+=1
                break
            row_num+=1
    return a,b
#@timer
def getXandYshifts (HP,planewave,aberrated,hp_img):
    # label=True
    # ps=0.01
    # shift=0.00000001
    #hp_img =86.71711691456846
    #hp_img= get_hp_img(planewave,num_of_holes)
    # print ('hp _ image ='+str(hp_img))
    #hp_img = 140/2.5*0.2*30
    a,b = getAandB(hp_img,HP)
    
    holes1= []
    for k in range(HP.num_of_holes):    
        holes1.append(Hole(hp_img,a[k],b[k]))
        
    xShifts1=np.zeros(HP.num_of_holes)
    yShifts1=np.zeros(HP.num_of_holes)    
    # plt.figure()
    # plt.imshow(imageTotal1)
    num=0
    # plt.figure()
    for hole in holes1:  
        hole.defineArea(planewave,int(hp_img))#size =140 for 37 hole 
        hole.calculateCenter()
        x1, y1 = hole.getCenters()
        xShifts1[num],yShifts1[num]=hole.getShifts()
        num=+num+1
        # if label:
        #     plt.plot((x1*ps)-shift,(y1*ps)-shift,'x',color='r',label='Planewave')
        #     label=False
        # else:
        #     plt.plot((x1*ps)-shift,(y1*ps)-shift,'x',color='r')
    
    holes2= []
    for k in range(HP.num_of_holes):    
        holes2.append(Hole(hp_img,a[k],b[k]))
    
    xShifts2=np.zeros(HP.num_of_holes)
    yShifts2=np.zeros(HP.num_of_holes)    
    num=0
    for hole in holes2:  
        hole.defineArea(aberrated,int(hp_img)) #size =140 for 37 hole 
        hole.calculateCenter()
        x2, y2 = hole.getCenters()
        # plt.plot(x2,y2,'o',color='y')
        x2, y2 = hole.getCenters()
        xShifts2[num],yShifts2[num]=hole.getShifts()
        num=+num+1
        # plt.plot(x2,y2,'o',color='y')
        
    xShifts = np.subtract(xShifts2, xShifts1)
    yShifts = np.subtract(yShifts2, yShifts1)
    
    # label=True
    # factor = 50/min(np.min(xShifts),np.min(yShifts))
    for k in range(HP.num_of_holes):  
        x1, y1 = holes1[k].getCenters()
        # if label:   
        #     plt.plot(((x1+xShifts[k]*factor)*ps)-shift,((y1+yShifts[k]*factor)*ps)-shift,'o',color='y',label='Distorted')
        #     label=False
        # else:
        #     plt.plot(((x1+xShifts[k]*factor)*ps)-shift,((y1+yShifts[k]*factor)*ps)-shift,'o',color='y')
    # plt.show()
    #xShifts,yShifts =generate_shifts('defocus',37)
    #xShifts,yShifts = 0,0
    # plt.axis('square')
    # plt.legend()
    # plt.xlabel('X (pixels)')
    # plt.ylabel('Y (pixels)')    
    return xShifts,yShifts
#@timer
def get_W (HP,xShifts,yShifts,L,pixelsize):
    holes_number=HP.num_of_holes
    #Use the calculated shifts to get dw/dx and dw/dy
    dWdX=np.zeros([holes_number])
    dWdY=np.zeros([holes_number])
    # DeltaX= 286.1
    # DeltaY=286.1
    # AB= 1750
    # DE=575
    for k in range (holes_number):
        dWdX[k]=xShifts[k]*pixelsize/L #/DeltaX)*(DE/AB)
        dWdY[k]=yShifts[k]*pixelsize/L #/DeltaY)*(DE/AB)
        
    #Create the spacing array and WF_diff
    Spacing =np.zeros([holes_number,holes_number], dtype = int)
    WF_diff=np.zeros([holes_number,holes_number])
    full_WF_diff=np.zeros([holes_number,holes_number])
    W=np.zeros([holes_number])
    for i in range (holes_number):
        for j in range (holes_number):
            full_WF_diff[i,j]=(((dWdX[i]+dWdX[j])/2) * (HP.HP_x[i]-HP.HP_x[j]))+\
                        (((dWdY[j]+dWdY[i])/2) * (HP.HP_y[i]-HP.HP_y[j]))
    for m in range (holes_number):
        for n in range (holes_number):
            if m ==n:
                continue
            elif (((HP.HP_x[m]-HP.HP_x[n])**2+(HP.HP_y[m]-HP.HP_y[n])**2)**0.5) -HP.hp <= HP.hs:
                Spacing[m,n]=1
                WF_diff[m,n]=(((dWdX[m]+dWdX[n])/2) * (HP.HP_x[m]-HP.HP_x[n]))+\
                (((dWdY[m]+dWdY[n])/2) * (HP.HP_y[m]-HP.HP_y[n]))


    #Iterate to get the wavefront W
    Error=np.zeros(500)
    Wprev=np.zeros([holes_number])
    # minerror=1.87e-135
    
    for k in range (500):
        for m in range (holes_number):
            sumAct =  np.sum(WF_diff[m,:])
            sumAct2 = np.sum(Spacing[m,:]*Wprev[:])
            sumAct3 = np.sum(Spacing[m,:])
            W[m] = (sumAct + sumAct2) / sumAct3
        W = W-np.mean(W)
        for i in range (holes_number):
            Error[k] = Error[k] + abs(W[i]-Wprev[i])
        for i in range (holes_number):
            Wprev[i]=W[i]
        # print(str(Error[k]))
        
    # plt.figure()   
    # plt.plot(Error[:k])
    # # plt.title("Error curve")
    # plt.xlabel('Number of iterations')
    # plt.ylabel('Error value')
    # plt.show() 
    return W
#@timer
class myColorBar():
    def __init__(self,thisW):
        self.minValue = np.min(thisW)
        self.maxValue = np.max(thisW)
    
    def getColor(self, thisW):
        lower = self.minValue
        upper = self.maxValue
        thisColor = plt.cm.RdBu_r((thisW-lower)/(upper-lower))
        return thisColor
#@timer
def plot_without_interpolation(HP,W,xref0=0,yref0=0,z=0):
    if np.all(xref0==0):
        font = {'family' : 'Arial', 'size'   : 15}
        
        plt.rc('font', **font)     
        plt.figure()

        mCB = myColorBar(W)
        for i in range(HP.num_of_holes):
            plt.scatter(HP.HP_x[i],HP.HP_y[i],color = mCB.getColor(W[i]), cmap="bwr")
        plt.axis('square')
        plt.xlabel('X [mm]')
        plt.ylabel('Y [mm]')
        plt.show()
        return
        
        
        
    font = {'family' : 'Arial',
    'size'   : 15}
    
    plt.rc('font', **font)     
    plt.figure()
    plt.pcolor(xref0,yref0,
    z,
    shading='nearest',
    cmap="RdBu_r"
    )
    plt.colorbar(label='[Waves]')
    mCB = myColorBar(W)
    for i in range(HP.num_of_holes):
        plt.scatter(HP.HP_x[i],HP.HP_y[i],color = mCB.getColor(W[i]), cmap="bwr")
    plt.axis('square')
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.show()

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

def toc(TicToc,tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print ( "Time taken: %f seconds.\n" %tempTimeInterval)

def tic(TicToc):
    # Records a time in TicToc, marks the beginning of a time interval
    toc(TicToc,False)
    
@timer
def plot_wavefront(xref0,yref0,z,title):
        font = {'family' : 'Arial',
        'size'   : 15}
        
        plt.rc('font', **font)     
        plt.figure()
        plt.pcolor(xref0,yref0,
        z,
        shading='nearest',
        cmap="RdBu_r"
        )
        plt.colorbar(label='[Waves]')
        plt.axis('square')
        plt.xlabel('X [mm]')
        plt.ylabel('Y [mm]')
        plt.title(title)
        plt.show()
        
#@timer
def plot_diffraction(exit_pupil_dia,diffraction,title):
    res=1024
    eps = 0.0001
    x=np.linspace(-1-eps,1+eps,res)*exit_pupil_dia
    y=np.linspace(-1-eps,1+eps,res)*exit_pupil_dia

    font = {'family' : 'Arial',
    'size'   : 15}
    
    plt.rc('font', **font)     
    plt.figure()
    plt.pcolor(x,y,
    diffraction,
    shading='nearest',
    cmap="RdBu_r"
    )
    plt.colorbar(label='Average no. electrons [1]')
    plt.axis('square')
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.title(title)
    plt.show()
        
# def pop_analysis (newPOP):

#@timer
def set_zernike(ZOSAPI,Surface_1, coeff,type_of_distortion):
    if type_of_distortion=='defocus':
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par13).IntegerValue = 5
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par15).DoubleValue = 0 #piston
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par16).DoubleValue = 0 #tip
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par17).DoubleValue = 0 #tilt
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par18).DoubleValue = coeff #defocus
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par19).DoubleValue = 0 #astigma
    elif type_of_distortion=='third_order':
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par13).IntegerValue = 7
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par15).DoubleValue = 0 #piston
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par16).DoubleValue = 0 #tip
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par17).DoubleValue = 0 #tilt
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par18).DoubleValue = 0 #defocus
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par19).DoubleValue = 0 #defocus
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par20).DoubleValue = coeff #defocus
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par21).DoubleValue = 0 #defocus
    elif type_of_distortion=='thermal':
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par13).IntegerValue = 56
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par15).DoubleValue = coeff[0] 
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par18).DoubleValue = coeff[1] 
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par25).DoubleValue = coeff[2]
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par36).DoubleValue = coeff[3]
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par51).DoubleValue = coeff[4]
        Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par70).DoubleValue = coeff[5]
    # elif type_of_distortion=="all37":
    #     textThis = "Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par"+str(13)+").IntegerValue = 37"
    #     # print(textThis)
    #     exec(textThis)
    #     Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par14).DoubleValue = thisApertureDia
    #     for dist_i in range(36):
    #         # textThis = "Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par"+str(dist_i+16)+").DoubleValue = coeff["+str(dist_i)+"]"
    #         textThis = "Surface_1.GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par"+str(dist_i+15)+").DoubleValue = coeff["+str(dist_i)+"]"
    #         # print(textThis)
    #         exec(textThis)
    

def reshape(data, x, y, transpose = False):
    """Converts a System.Double[,] to a 2D list for plotting or post processing
    
    Parameters
    ----------
    data      : System.Double[,] data directly from ZOS-API 
    x         : x width of new 2D list [use var.GetLength(0) for dimension]
    y         : y width of new 2D list [use var.GetLength(1) for dimension]
    transpose : transposes data; needed for some multi-dimensional line series data
    
    Returns
    -------
    res       : 2D list; can be directly used with Matplotlib or converted to
                a numpy array using numpy.asarray(res)
    """
    if type(data) is not list:
        data = list(data)
    var_lst = [y] * x;
    it = iter(data)
    res = [list(islice(it, i)) for i in var_lst]
    if transpose:
        return transpose(res);
    return res
#@timer
def extract_data(analysis):
    analysis.ApplyAndWaitForCompletion()
    Results = analysis.GetResults()
    InterferenceGrid = Results.GetDataGrid(0).Values
    interference = reshape(InterferenceGrid, InterferenceGrid.GetLength(0),InterferenceGrid.GetLength(1))
    interference=np.array(interference)
    return interference
    
def excel_save(file_name,fom):
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    row = 0
    
    for col, data in enumerate(fom[0,:,:]):
        worksheet.write_column(row, col, data)
    workbook.close()
    
@timer
def get_fom(W,wref,HP):
    print('Getting Standard deviation between W and wref...')
    n= HP.num_of_holes
    wf_diff=0
    for i in range(n):
        wf_diff= wf_diff+ (W[i]-wref[i])**2
    standard_deviation=np.sqrt(wf_diff/(n*1.))
    figure_of_merit= 100-(standard_deviation/(np.max(W)-np.min(W)))*100
    print('Standard dev = '+str(standard_deviation) + ' FOM = ' + str(figure_of_merit))
    return standard_deviation,figure_of_merit

#@timer
def get_discrete_points(xref,yref,wavefront,HP,res):
    wNotNan=[]
    xref2,yref2=[],[]
    for i in range(res):
        for j in range(res):
            if(math.isnan(wavefront[i,j])):
                continue
            wNotNan.append(wavefront[i,j])
            xref2.append(xref[i,j])
            yref2.append(yref[i,j])    
    z3=SmoothBivariateSpline(xref2,yref2,wNotNan)#kind='linear'
    wref=np.zeros(HP.num_of_holes)   
    for i in range(HP.num_of_holes):
        wref[i] = z3(HP.HP_x[i], HP.HP_y[i])
    wref-=np.min(wref)        
    return wref
   
def poisson(irradianceExact,div):
    NeMax = 450000 #*(1.6*10e-19)
    QE = 0.5 # quantum efficiency (incident photons/electrons) 
    maxIrradiance = np.max(irradianceExact)    
    Ne = NeMax * (irradianceExact/maxIrradiance) # number of electrons at each pixel
    Np = Ne/QE 
    shot_noise=np.random.normal(Np,np.sqrt(Np)/div, size=[len(irradianceExact),len(irradianceExact)])
    Ne_new = shot_noise*QE
    irradianceExact_new = Ne_new/NeMax*maxIrradiance
    return irradianceExact_new

""" This scrip generates a modelled ice volume. The volume is a discrete voxel
space with regions assigned as either boundaries or interiors based on a 
Voronoi tesselation. Each voxels is assigned an intensity value, a float value
that represents the amount of some impurity present at that point in space, from
a probability distriubtion which originates from input LA-ICP-MS data. To 
generate the volume, input relevant data by either providing the path to a 
JSON file  (calling data_From_File()), or input the data directly, 
(calling data_From_IDE()). Once run, this data can be analysed by running
_02_Analyse.py """

import time
import iceImpurityFramework as IIF
import numpy as np
from matplotlib import pyplot as plt
import os
import json
start_time = time.time()
np.random.seed(IIF.get_Seed())  # set the seed to a fixed value

def generate_LA_Model(iceName, targetGrainRad, iceSize, chemPath, impurityName, 
                      threshold, voxelSize, maskPath):
    """
    Utilising IFF module, this function generates the modelled ice volume. It
    can take a VERY long time to run
    paths are relative to Model_Inputs/{ice name}/
    """ 
    # Get saved data
    chemicalData = np.load(os.path.join('Model_Inputs/',
                                        iceName, f'{chemPath}.npy'))
    segmentationData = np.load(os.path.join('Model_Inputs/',
                                            iceName, f'{maskPath}.npy'))
    # Ensure mask is binary
    segmentationData = IIF.make_Mask(segmentationData, 50)
    fig = plt.figure()
    origAx = fig.add_subplot(1,1,1) 
    origAx.imshow(chemicalData, cmap='hot', interpolation='nearest')
    # Make and save model
    IIF.LAModel(iceName, targetGrainRad, iceSize, chemicalData, impurityName, 
                          threshold, voxelSize, segmentationData)
    return 0

def data_From_IDE():
    """ Function used to pass parameter, which are contained in a json file,
    to model generation function -  usefull for managing multiple experimental
    inputs which are run at different times """
    name = 'EDC_514_1_20231115' # [str] Name to save model under   
    average_Grain_Radius = 500  # Target modelled grain radius - micrometers
    ice_Dimensions = [4000, 4000, 4000] # Target dimension x,y,z in micrometers
    chemical_File = 'chemical' # Chemical data file name
    chemical_Name = "Sodium" # Name of modelled impurity - only used for labels
    chemical_Threshold = 1000000 # Intensity threshold for 2d VISUALISATIONS
    chemical_Pixel_Size = 40 # LAICPMS spot size used to collect chemical data
    mask_File = 'mask' # Name of mask file 
    generate_LA_Model(name, average_Grain_Radius, ice_Dimensions, chemical_File, 
                      chemical_Name, chemical_Threshold, chemical_Pixel_Size, 
                      mask_File)
    
    return 0

def data_From_File(filePath):
    """ Function used to pass parameters, which are contained in a json file,
    which is saved at the file path passed to the function, to model generation 
    function. This approach is usefull for managing experimental inputs as ice 
    data is saved, and can be re-used"""
    with open(filePath, 'r') as file:
        iceData = json.load(file)
    generate_LA_Model(iceData["name"],
                      iceData["average_Grain_Radius"],
                      iceData["ice_Dimensions"],
                      iceData["chemical_File"], 
                      iceData["chemical_Name"], 
                      iceData["chemical_Threshold"],
                      iceData["chemical_Pixel_Size"],
                      iceData["mask_File"])
    return 0

if __name__ == '__main__':
    #data_From_IDE()
    filePath = "Model_Inputs/Test/ice_data.JSON"
    data_From_File(filePath)







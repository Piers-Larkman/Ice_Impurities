""" This script creates plots illustrating framework operation"""

import iceImpurityFramework as IIF
import numpy as np
from matplotlib import pyplot as plt
import os
import json
np.random.seed(IIF.get_Seed())  # set the seed to a fixed value

def generate_LA_Model(iceName, targetGrainRad, iceSize, chemPath, impurityName, 
                      threshold, voxelSize, maskPath, boundariesBool = 1, p = 3):
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
                          threshold, voxelSize, segmentationData, 
                          boundariesBool = boundariesBool, p = p)
    return 0


def data_From_File(filePath, boundariesBool = 1, p=3):
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
                      iceData["mask_File"],
                      boundariesBool,
                      p)
    return 0

if __name__ == '__main__':
    filePath = "Model_Inputs/Test/ice_data.JSON"
    """ Create model """
    data_From_File(filePath) # Generate standard model
    #data_From_File(filePath, boundariesBool = 0) # Generate model with no boundaries
    #data_From_File(filePath, boundariesBool = 0, p = 1) # Change distance metric    
    #Test_Sodium_[400, 800, 1200]_[10, 20, 30]_40Mics
    import sys
    sys.exit()
    """ Make plots from model """
    fullModelPath = "Example_[400, 800, 1200]_[10, 20, 30]_P3_Bound"
    p3NoboundPath = "Example_[400, 800, 1200]_[10, 20, 30]_P3_noBound"
    p1NoboundPath = "Example_[400, 800, 1200]_[10, 20, 30]_P1_noBound"

    fullModel = IIF.pickle_Load(f"Models/{fullModelPath}/Data/Space.pkl")
    p3Nobound = IIF.pickle_Load(f"Models/{p3NoboundPath}/Data/Space.pkl")
    p1Nobound = IIF.pickle_Load(f"Models/{p1NoboundPath}/Data/Space.pkl")

    if not os.path.exists("Example_Plots"):
        os.makedirs("Example_Plots")

    # Create figure and axes objects
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')

    ax1.text2D(0.5, 0.8, 'Grain representation', ha='center', va='center', transform=ax1.transAxes, fontsize = 30)
    ax2.text2D(0.5, 0.8, 'Intensity representation', ha='center', va='center', transform=ax2.transAxes, fontsize = 30)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax4.set_axis_off()

    fullModel.plot_Cell(0, ax = ax1)
    p1Nobound.plot_Cell(0, ax = ax2)
    p3Nobound.plot_Cell(0, ax = ax3)
    fullModel.plot_Cell(0, ax = ax4)

    #fullModel.plot_Cell(0, mode = "intensity", ax = ax2)
    
    plt.tight_layout()
    plt.savefig('Example_Plots/representations.png', bbox_inches='tight', dpi=300)
    
    """
    # Create figure and axes objects
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.text2D(0.5, 0.8, 'Example grain', ha='center', va='center', transform=ax1.transAxes, fontsize = 30)
    ax2.text2D(0.5, 0.8, 'Example boundary', ha='center', va='center', transform=ax2.transAxes, fontsize = 30)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0)
    ax1.set_axis_off()
    ax2.set_axis_off()
    
    fullModel.plot_Feature(0, featInd = 1, featureType = "grain", labels = 1, ax = ax1)
    fullModel.plot_Feature(0, featInd = 4, featureType = "boundary", labels = 1, ax = ax2)
    """
    plt.tight_layout()
    plt.savefig('Example_Plots/features.png', bbox_inches='tight', dpi=300)
    
    # Model is constructed, make plots with it
    """
    print("Plotting 3d rep")        
    IIF.pickle_Load(self.spaceFilePath)
    coords = [[0, interrogator.spaceSize[1], 0]]
    space = interrogator.load_Space()
    cellIDs = space.cellsWithCoords(coords)
    interrogator.plot_3d_Reps(cellIDs[0], save = 1)
    #show_3d_Loc_on_Face(coords[0], space.cellSize, [intenseFaceAx, grainFaceAx])
    """






















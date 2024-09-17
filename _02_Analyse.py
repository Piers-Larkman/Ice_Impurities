""" This scrip accesses relevant information generated by a modelled ice
volume and runs an analysis routine, as such it should be run after
 _01_generateCells.py. """

import matplotlib.pyplot as plt
import numpy as np
import iceImpurityFramework as IIF
import time
import os
from scipy.ndimage import gaussian_filter1d
import datetime
import pandas as pd

# --- Experimental functions --- #
def graph_Lines(linesData, ax, column = '[23Na]+', colours = IIF.get_Colours(),
                legendTitle = None, labels = [None, None], xLabel = 1, 
                yLabel = 1, smoothing = 0, log = 1):
    for i, lineData in enumerate(linesData):
        xs = lineData['# X pos [um]'].astype(float)/10000            
        ys = lineData[column].astype(float)
        if smoothing != 0:
            ys = gaussian_filter1d(ys, smoothing)
        ax.plot(xs, ys, label = labels[i], color = colours[i])
    ax.set_xlim(min(xs), min(max(xs), 80000))  # Set x-axis limits based on the data
    IIF.axis_Visuals(ax, xLabel = xLabel, yLabel = yLabel, 
                     legendTitle = legendTitle, scale = " cm ", log = log)
    return 0

def get_Big_Spot(arr, profileCenter, spotPadding):
    """
    Operating on 2d array, return the resulting profile of a larger spot size
    currently only odd-sized spots.
    """
    size = spotPadding*2 + 1
    # Mean horizontally
    xVals = model_zVals(len(arr[:,0]), voxelSize)
    lateralAveraged = np.mean(arr[:,profileCenter-spotPadding:profileCenter+spotPadding], axis =1)
    # Mean vertically and reduce resolution
    intensity = []
    newXs = []
    for i in range(0, len(lateralAveraged), size):
        segment = lateralAveraged[i:i+size]
        xSeg = xVals[i:i+size]
        if len(segment) == size:
            intensity.append(np.mean(segment))
            newXs.append(np.mean(xSeg))
    return intensity, newXs

# ---- Model functions ---- #
def plot_Profiles_For_Paper(array, cfa, referenceValue,
                    refInds, cfaSmoothing, save = 0):
    # Plot profiles with intensities normalised to the average intensity in
    # the space and passed through a gaussian filter with width of laser spot size

    fig, axs = plt.subplots(5, 1, gridspec_kw={'hspace': 0.02})
    fig.set_size_inches(figHeight, figWidth)
    fig.suptitle(f'Modelled {iceName} profile signals', fontsize = 36)
    plt.subplots_adjust(top=0.9)
    
    """
    # Plot some profiles with reference intensity line
    for ax in axs:
        ax.axhline(y=referenceValue, color="#CC79A7", linewidth = 2, label=r'Volume average')
    """
    
    profile = gaussian_filter1d(array[:,refInds[0]]/referenceValue, 
                                1, mode='reflect')
    axs[0].plot(profile, label = "Signal from profile 1", color = "k", linewidth = 1)
    profile = gaussian_filter1d(array[:,refInds[1]]/referenceValue, 
                                1, mode='reflect')
    axs[0].plot(profile, label = "Signal from profile 2", color = "#0072B2", linewidth = 1)
    axs[0].set_ylim(0.001, 50)

    # Larger spot size
    extra = 1
    
    intenisty, x = get_Big_Spot(array, refInds[0], extra)
    intenisty = gaussian_filter1d(intenisty, 1+extra*2, mode='reflect')
    axs[1].plot(x, intenisty/referenceValue, label = "Signal from profile 1", color = "k", linewidth = 1)
    intenisty, x = get_Big_Spot(array, refInds[1], extra)
    intenisty = gaussian_filter1d(intenisty, 1+extra*2, mode='reflect')
    axs[1].plot(x, intenisty/referenceValue, label = "Signal from profile 2", color = "#0072B2", linewidth = 1)
    axs[1].set_xlim(min(x), max(x))

    # Full face combination, unsmoothed
    selectedProfiles = array[:, :]
    fullProfile = np.mean(selectedProfiles, axis=1)
    fullProfile = gaussian_filter1d(fullProfile, 1, mode='reflect')
    axs[2].plot(fullProfile/referenceValue, label = "Face spatially averaged signal \nunsmoothed", color = "#009E73", linewidth = 1)

    # Plot on different scale with CFA
    fullProfileSmoothed = gaussian_filter1d(fullProfile, cfaSmoothing)
    axs[3].plot(fullProfileSmoothed/referenceValue, label = f"Face spatially averaged signal \nσ = {int(cfaSmoothing*voxelSize)} µm", color = "#009E73", linewidth = 2)
    axs[4].plot(cfa/referenceValue, label = "Simulated CFA", color = "#E69F00", linewidth = 2)  
    
    fig.text(0.08, 0.5, 'Normalised sodium intensity', ha='center', va='center', fontsize=36, rotation = 90)
    IIF.axis_Visuals(axs[0], xLabel = 0, yLabel = 0, legendTitle = f"Spot size: {voxelSize} µm \nLateral separation: {np.abs(refInds[1]-refInds[0])*voxelSize} µm", log = 1) 
    IIF.axis_Visuals(axs[1], xLabel = 0, yLabel = 0, legendTitle = f"Spot size: {voxelSize*(extra*2+1)} µm", log = 1, scale = None) 
    IIF.axis_Visuals(axs[2], xLabel = 0, yLabel = 0, legendTitle = f"Spot size: {voxelSize} µm", log = 0) 
    IIF.axis_Visuals(axs[3], xLabel = 0, yLabel = 0, legendTitle = f"Spot size: {voxelSize} µm", log = 0) 
    IIF.axis_Visuals(axs[4], xLabel = 1, yLabel = 0, legendTitle = None, log = 0) 

    # Amend axs and labels
    for i, ax in enumerate(axs):
        if i != 1:
            model_zax(ax, len(fullProfile), voxelSize, setlims = 1)
        if i == len(axs)-1:
            pass
        else:
            ax.set_xticklabels([])

        ax.text(0.02, 0.95, f'({chr(97+i)})', transform=ax.transAxes,
                fontsize=28, va='top', ha='left')
    
    return fig


def get_Mean_Max_Min(xs, ys):
    # Create a dictionary to store y values for each x value
    uniqueXs = {}

    # Accumulate y values for each unique x value
    for x, y in zip(xs, ys):
        if x in uniqueXs:
            uniqueXs[x].append(y)
        else:
            uniqueXs[x] = [y]

    # Calculate the mean of y values for each unique x value
    xSubset = []
    yMeans = []
    yMins = []
    yMaxs = []
    for x, ys in uniqueXs.items():
        xSubset.append(x)
        yMeans.append(sum(ys) / len(ys))
        yMins.append(max(ys))
        yMaxs.append(min(ys))

    return xSubset, yMeans, yMins, yMaxs

def first_Below(lst, thresh):
    for i, value in enumerate(lst):
        if value < thresh:
            return i
    return None  # If threshold is not reached

def find_sign_change_index(arr):
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            return i
    return -1  # If no sign change is found

def find_Thresh_X(x,y):
    from scipy.signal import savgol_filter
    x=np.array(x)
    y=np.array(y)
    # Apply Savitzky-Golay filtering to smooth the data
    window_size = 11
    poly_order = 2
    y_smooth = savgol_filter(y, window_size, poly_order)
    
    # Compute the first and second derivatives of the smoothed data
    dy_dx = np.gradient(y_smooth, x)
    d2y_dx2 = np.gradient(dy_dx, x)
    
    # Find the inflection point by locating where the second derivative changes direction
    inflection_indices = find_sign_change_index(d2y_dx2)
    
    inflection_point = x[inflection_indices]
    return inflection_point

def plot_MAD_For_Paper(interrogator, interestingNumProfs, save = 1):
    # Prepare parameters
    colours = ["#009E73", "k", "#0072B2" , "green"]
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(figHeight, figWidth)
   
    bestCases = [] # store [spotSize, smoothing, minMAD]
    experimentalCases = [] # store [spotSize, smoothing, numProfiles, MAD]
    profilesBelowLim = [] # store [spotSize, smoothing, numProfiles]
    profilesForReduction = [] # store [spotSize, smoothing, reduction, numProfiles]
    MADObjs = interrogator.MADObjects

    for i, spotSize in enumerate(interrogator.MADspotSizes):
        for j, smoothing in enumerate(interrogator.smoothingsForMAD[i]):
                xs = []
                ys = []
                minMAD = float('inf')
                # Select desired MAD ojbects
                for obj in MADObjs:
                    if obj.smoothing == smoothing and obj.spotSize == spotSize:
                        ys.append(obj.MAD)
                        xs.append(len(obj.profileIDs))
                # Get mean, min, max values for plot for each num of profiles
                xs, yMean, yMin, yMax = get_Mean_Max_Min(xs, ys)
                minMAD = min(minMAD, min(yMean))
                label = f"σ = {int(smoothing*spotSize*voxelSize)} µm"
                axs[i].plot(xs, yMean, color = colours[j], label = label)
                axs[i].fill_between(xs, yMax, yMin, color=colours[j], alpha=0.2)
                # Add indicator where MAD drops below 20% for each line
                thresh = 20
                indexBelow20 = first_Below(yMean, thresh)
                # Plot a horizontal line at the threshold level
                """
                if indexBelow20 != None:
                    # Plot a vertical line at the x value where y drops below the threshold
                    axs[i].axvline(x=xs[indexBelow20], color='r', linestyle='--', label=f'{xs[indexBelow20]:.0f} profiles', ymax = thresh / axs[1].get_ylim()[1])
                """
                # Save cases reported in paper
                bestCases.append([spotSize, smoothing, minMAD])
                if indexBelow20 == None:    
                    profilesBelowLim.append([spotSize, smoothing, None])
                else:
                    profilesBelowLim.append([spotSize, smoothing, xs[indexBelow20]])
                if spotSize == 1:
                    ind = xs.index(1)
                    experimentalCases.append([1, smoothing, 1, yMean[ind]])
                    ind = xs.index(interestingNumProfs)
                    experimentalCases.append([1, smoothing, interestingNumProfs, yMean[ind]])
                """
                if j == 0:
                    # Calc and plot elbow point (based on largest smoothing)
                    elbow_index = find_Thresh_X(xs, yMean)
                    axs[i].axvline(x=xs[elbow_index], color='yellow', linestyle='--', label='Elbow Point')
                    print(xs[elbow_index])
                """
                # Calculate number of profiles required to access a certain
                # reduction in initial MAD
                # Find approx num profiles to reduce MAD by some factor
                for factor in [2,10]:
                    maxVal = max(yMean)
                    target = maxVal/factor
                    numProfs = xs[np.abs(yMean - target).argmin()]
                    profilesForReduction.append([spotSize, smoothing, factor, numProfs])

        axs[i].axhline(y=thresh, color='r', linestyle='--', label='Example upper threshold')
        axs[i].legend(title = f"Spot size: {spotSize*voxelSize} µm", fontsize = 28, title_fontsize=32, loc = "upper right")

    for i, ax in enumerate(axs):
        # Adjusting tick parameters and grid appearance
        ax.tick_params(axis='both', which='major', labelsize=ticksSize)  # Increase font size of ticks
        ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.5)  # Faint grid lines
        #ax = brokenaxes(xlims=((-5, interrogator.spaceSize[1]+5), (crossSec-20, crossSec+20))) 
        #ax.scatter(crossSec, CFAMAD, color="#40E0D0", label=r'CFA Pd')
        ax.set_ylabel('MAD (%)', fontsize=labelsSize, labelpad=20)
        ax.set_xlabel('Number of profiles combined', fontsize=labelsSize, labelpad=20)  # Set X axis label with larger font size
        ax.text(0.02, 0.95, f'({chr(97+i)})', transform=ax.transAxes,
                fontsize=28, va='top', ha='left')
    axs[0].set_xlim(0, 100)
    axs[1].set_xlim(0, 80)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.suptitle(f'Modelled {iceName}', fontsize = 36)

    
    if save == 1:
        filepath = 'Outputs\\Plots\\Graphs\\parameterSpace.png'
        IIF.check_Create_Filepath(filepath)
        plt.savefig(filepath, bbox_inches='tight')
    
    
    return (bestCases, experimentalCases, profilesBelowLim, 
            interrogator.CFAMAD/interrogator.volumeIntensity[0], profilesForReduction)

def plot_Prob_Dist(ax = None):
    """
    Plot boundary and interior grain intensity probability distributions.
    Assumes probs are stored as [interiorDist, boundaryDist]
    """
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    # Load probability data
    probs = IIF.pickle_Load("Outputs/Interim/Empirical/probs.pkl")
    # Plot frequency histogram and probability plots
    IIF.plot_Probdist(probs[0], ax, label = "Interior ", colour = "#FF00FF")
    IIF.plot_Probdist(probs[1], ax, label = "Boundary ", colour = "#40E0D0")
    return 0

def plot_Face(arr, spotSize, profiles = None, save = 0, overlay = 0, nameAddition = "", cmap = "inferno"):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(figHeight, figWidth)
    arr = np.transpose(arr)
    arr = np.flip(arr, axis=0)
    
    if overlay == 1:
        overlay_Profiles(ax, profiles)
    
    ax.imshow(arr, cmap= cmap, interpolation='none')
    model_zax(ax, arr.shape[1], spotSize*voxelSize)
    face_xax(ax, arr.shape[0], spotSize*voxelSize)
    plt.xlabel('Relative model depth (cm)', fontsize=36)
    plt.ylabel('Lateral position (cm)', fontsize=36)
    fig.text(0.91, 0.5, f'Modelled {iceName}', ha='center', va='center', fontsize=36, rotation = -90)
    if save == 1:
        filepath = f'Outputs\\Plots\\Visuals\\oneFace{nameAddition}.png'
        IIF.check_Create_Filepath(filepath)
        plt.savefig(filepath, bbox_inches='tight')
        filepath = f'Outputs\\Interim\\Model\\oneFace{nameAddition}'
        IIF.pickle_Save(filepath, arr)
    return ax

def overlay_Profiles(ax, profiles):
    # Extract y locations
    colours = ["blue", "red", "#FF00FF", "#40E0D0"]
    for q, profile in enumerate(profiles):        
        # Plot as line on imshow
        ax.axhline(y=profile.yPos, color=colours[q], linewidth=2)
    return 0

#TO DO - Possibly MAKE BOTH OF THESE PART OF IIF classes

def model_zax(ax, axLen, spotSize, setlims = 0):
    maxVal = int(np.floor(axLen*spotSize/10000))
    xVals = np.linspace(0, maxVal, maxVal+1)
    xVals = xVals.astype(int)
    positions = xVals/(spotSize/10000)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(val) for val in xVals], fontsize=labelsSize)
    if setlims == 1:
        ax.set_xlim(0, axLen)
    return xVals

def model_zVals(axLen, spotSize):
    return list(range(0, axLen*spotSize + 1, spotSize))[:-1]

def face_xax(ax, axLen, spotSize):
    maxVal = int(np.floor(axLen*spotSize/10000))
    xVals = np.linspace(0, maxVal, maxVal*2+1)
    positions = (xVals/(spotSize/10000))[::-1]
    ax.set_yticks(positions)
    ax.set_yticklabels([str(val) for val in xVals], fontsize=labelsSize)
    return 0

def read_Experimental():
    # Read lines data - already pre-processed (corrected, zeros clipped)
    linesFilePath = f"Model_Inputs/{dataName}/profiles.csv"
    df = pd.read_csv(linesFilePath)
    header = df.columns.tolist()  # Get the list of column names as the header
    df = df[~df.apply(lambda row: row.tolist() == header, axis=1)]  # Drop rows that match the header
    df = df.apply(pd.to_numeric, errors='coerce')
    # Create some new columns in the df.
    # Create a total combined signal - sum x values and allow for some float imprecision
    # Note that during experimental data collection, when line focus height is changed
    # the laser appears to fires twice
    df['[23Na]+'] = pd.to_numeric(df['[23Na]+'], errors='coerce')
    df['# X pos [um]'] = pd.to_numeric(df['# X pos [um]'], errors='coerce')
    df['# X pos [um]'] = (df['# X pos [um]'] //voxelSize)*voxelSize
    # Identify repeated values and calculate mean values
    combinedDf = df.groupby('# X pos [um]').agg({'[23Na]+': ['mean', 'count']}).reset_index()
    combinedDf.columns = ['# X pos [um]', '[23Na]+', 'num profiles combined']
    return df, combinedDf

def plot_Experimental_Lines(experimentalDf, combinedProfilesDf):
    # Plot two lines
    fig, axs = plt.subplots(3, 1, gridspec_kw={'hspace': 0.02})
    fig.set_size_inches(figHeight, figWidth)
    
    # Plot some lines, first get
    uniqueYs = experimentalDf["Y pos [um]"].unique()
    
    profileIndexes = [3,1]
    profilesSeparation = np.abs(float(uniqueYs[profileIndexes[0]]) - float(uniqueYs[profileIndexes[1]]))
    profilesSeparationstr = "{:.0f}".format(profilesSeparation)
    
    graph_Lines([experimentalDf[experimentalDf["Y pos [um]"] == uniqueYs[profileIndexes[0]]], 
                experimentalDf[experimentalDf["Y pos [um]"] == uniqueYs[profileIndexes[1]]]], axs[0],
                legendTitle = f"Lateral separation: {profilesSeparationstr} µm",
                labels = ["Signal from profile 1", "Signal from profile 2"], colours = ["#0072B2", "k"], xLabel = 0, yLabel = 0)
    
    numProfs = (combinedProfilesDf['num profiles combined'].mode()[0]).astype(int)
    graph_Lines([combinedProfilesDf], axs[1],
                legendTitle = f"",
                labels = [f"Combination of {numProfs} profiles"], colours = ["#009E73"], yLabel = 0)
    
    smoothing = 10000/voxelSize
    graph_Lines([combinedProfilesDf], axs[2],
                legendTitle = f"",
                labels = [f"Combination of {numProfs} profiles \nσ = {int(smoothing*voxelSize)} µm"],
                colours = ["#009E73"], yLabel = 0, smoothing = smoothing, log = 0)
    
    fig.text(0.05, 0.5, 'Sodium intensity (counts)', ha='center', va='center', fontsize=36, rotation = 90)

    
    # Plot data and add labels
    for i, ax in enumerate(axs):
        ax.text(0.05, 0.95, f'({chr(97+i)})', transform=ax.transAxes,
                fontsize=26, va='top', ha='left')
    axs[0].set_xticklabels([])
    fig.suptitle(f'Experimental {iceName} profile signals', fontsize = 36)

    """
    fig.suptitle(f'Experimental {iceName} profiles', fontsize = 36)
    filepath = f'Empirical_Outputs/{dataName}/Modelled{iceName}single_combined_profiles'
    plt.savefig(filepath, bbox_inches='tight')
    """
    return profilesSeparation, numProfs

def main(model, initInterro = 0):
    # ---- Experimental data ---- #
    if dataName:
        experimentalDf, combinedProfilesDf = read_Experimental()
        profilesSeparation, numExperimentalProfs = plot_Experimental_Lines(experimentalDf, combinedProfilesDf)
    else:
        profilesSeparation = 120
        numExperimentalProfs = 10
        
    # ---- Modelled data ---- # 
    os.chdir(f"Models/{modelName}")
    start_time = time.time()
    interroFile = "Data/Interrogator.pkl"
    # Load analysis object
    interrogator = IIF.pickle_Load(interroFile)
    if initInterro == 1:
        interrogator.extract_Info()
        interrogator.generate_MAD(numExperimentalProfs)
        IIF.pickle_Save('Data/Interrogator.pkl', interrogator)
    
    faceIntensityArray = interrogator.faceIntensityArray
    faceGrainArray = interrogator.faceGrainArray
    faceProfiles = interrogator.faceProfiles
    
    meanIntensity = interrogator.volumeIntensity[0]
    
    # Establish some profile ID to be used as a reference
    referenceProfile = 40 

    profileInds = [referenceProfile, int(referenceProfile+profilesSeparation/voxelSize)]  
    referenceProfiles = []
    for ind in profileInds:
        referenceProfiles.append(faceProfiles[ind])


    print("Plotting profiles")        
    print("--- %s seconds ---" % (time.time() - start_time))    


    # If profiles generated during interrogator initalisation
    #profiles = interrogator.load_Profiles(duringGrowth = 0)
    # If profiles generated during model construction
    # Take two profiles
    #plot_Correlations_For_Paper(profiles, smoothing = 5, save = 1, load = 1)
    """
    print("Plotting grain and intensity information")
    print("--- %s seconds ---" % (time.time() - start_time))    
    interrogator.plot_Grain_Sizes(ax = "new", typeToPlot = "all", normaliseSize = 1, save = 1) 
    interrogator.plot_Grain_Sizes(ax = "new", typeToPlot = "interior", normaliseSize = 1, save = 1)
    """
    print("Plotting faces")        
    print("--- %s seconds ---" % (time.time() - start_time))

    """
    intenseFaceAx = plot_Face(faceIntensityArray, spotSize = 1, 
                    profiles = referenceProfiles, save = 1, overlay = 1, nameAddition = "_intensity", cmap = "viridis")

    grainFaceAx = plot_Face(faceGrainArray, spotSize = 1, 
                    profiles = referenceProfiles, save = 1, overlay = 1, nameAddition = "_grain", cmap = "viridis")
    """

    print("Performing and plotting face parameter space analysis")        
    print("--- %s seconds ---" % (time.time() - start_time))
    bestMADS, experimentMads, profilesBelowThresh, CFAMAD, reductionVals = plot_MAD_For_Paper(interrogator, numExperimentalProfs)
    
    plot_Profiles_For_Paper(faceIntensityArray, interrogator.smoothedCFASig, meanIntensity,
                            profileInds, interrogator.cfaSmoothing, save = 1)
    
    # Write some data to file
    with open(r'Outputs/modelInfo.txt', 'w') as file:
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M")
        # Write information with variables to the file
        file.write("Analysis information\n")
        file.write(f"Date and Time: {formatted_datetime}\n")
        file.write(f"Mean input average intensity: {np.mean(model.chemicalImage)}\n")
        file.write(f"Average voxel intensity: {meanIntensity}\n")
        file.write(f"Total number of boundary voxels: {interrogator.numBoundaryVoxels}\n")
        file.write(f"Total number of interior voxels: {interrogator.numInteriorVoxels}\n")
        file.write(f"Ratio gb/int: {interrogator.numBoundaryVoxels/interrogator.numInteriorVoxels}\n")
        file.write(f"MAD for CFA (%): {CFAMAD}\n")
        file.write("Best MADs (%) in format [spotSize, smoothing, lowest MAD (i.e MAD of max. num. profiles)]'\n'")
        # Iterate over each list in the list of lists
        for subset in bestMADS:
            # Write the sublist to the file as a string
            file.write(str(subset) + '\n')
        file.write("Experimental MADs in format [spotSize, smoothing, numProfiles, MAD]'\n':")
        for subset in experimentMads:
            file.write(str(subset) + '\n')
        file.write("Number of profiles below 20% MAD in format [spotSize, smoothing, numProfiles]'\n':")
        for subset in profilesBelowThresh:
            file.write(str(subset) + '\n')
        file.write("Number of profiles required for a relative reduction in MAD [spotSize, smoothing, reduction, numProfiles]'\n':")
        for subset in reductionVals:
            file.write(str(subset) + '\n')
            
        file.write(f"Upper cell size set to: {IIF.get_Upper_Cell_Size_Voxels()}\n")
        file.write(f"Seed number used for generation: {IIF.get_Seed()}")
    
    """
    #print("Plotting probability distriubtions")
    #plot_Prob_Dist()
    
    print("Plotting 3d rep")        
    print("--- %s seconds ---" % (time.time() - start_time))
    
    coords = [[0, interrogator.spaceSize[1], 0]]
    space = interrogator.load_Space()
    cellIDs = space.cellsWithCoords(coords)
    #interrogator.plot_3d_Reps(cellIDs[0], save = 1)
    #show_3d_Loc_on_Face(coords[0], space.cellSize, [intenseFaceAx, grainFaceAx])
    """
    
    return 0
    

if __name__ == '__main__':
    
    # Aspect ratio of A4 page
    labelsSize = 36
    ticksSize = 30
    pageAspectRatio = 8.27 / 11.69
    figWidth = 7  # inches
    figHeight = figWidth / pageAspectRatio
    modelName = "RECAP901_3_20230512_Sodium_[10000, 20000, 80000]_[280, 520, 2000]_40Mics"
    model = IIF.pickle_Load(f"Models/{modelName}/Model.pkl")
    # Make voxel size and ice name available as global variables
    voxelSize = model.voxelSize
    
    if modelName.startswith("RECAP976"):
        iceName = "RECAP LGP"
        dataName = "RECAP976_7_20230511"
    elif modelName.startswith("RECAP901"):
        iceName = "RECAP Holocene"
        dataName = "RECAP901_3_20230512"
    elif modelName.startswith("EDC_1994"):
        iceName = "EDC LGP"
        dataName = "EDC_1994_3_20231114"
    elif modelName.startswith("EDC_514"):
        iceName = "EDC Holocene"
        dataName = "EDC_514_1_20231115"
    else:
        iceName = "Unspecified ice"
        dataName = None
    
    main(model, initInterro = 0)











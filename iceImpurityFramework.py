"""
iceImpurityFramework V1.0 - This code forms the basis for generating and
analysing a modelled ice volume populated with an impurity distribution taken
from experimental LA-ICP-MS analysis.

Experimental LA-ICP-MS data are input into a Model class, which then
facilitates the generation (using code _01_generateCells) of a 3D Voronoi
tessellation and its analysis (using code _02_Analyse)
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import cdist 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#import multiprocessing
import random
import itertools
import math
import time
from scipy.ndimage import gaussian_filter1d
from math import factorial

start_time = time.time()

pageAspectRatio = 8.27 / 11.69
figWidth = 7  # inches
figHeight = figWidth / pageAspectRatio
    
class Model:
    """
    Class of the whole model, containing information on data input, the 
    generate space, and analysis carried out. Does not get directly instanced,
    but should instance sub-class. Currently only LAModel subclass is possible,
    but this leaves room for further development.
    
    Attributes:
        iceName (str): Name of ice being modelled
        filePath (str): Path to save/load model data to/from
        impurityName (str): Name of target impurity, only relevant to labelling
        spaceFilePath (str): Path to pickled space object
        interroFilePath (str): Path to picked interrogator object
        voxelSize (int): List of voxel objects contained in this grain
        iceSize (list): Actual (target) dimensions of ice in micrometers [x,y,z]
        spaceSize (list): Dimensions of modelled space in voxels [x,y,z] 
        targetGrainRad (int): Target grain radius in micrometers
    """
    def __init__(self, iceName, targetGrainRad, voxelSize, impurityName, 
                 iceSize, spaceSize):
        """        
        Arguments:
            iceName (str): Name of ice being modelled
            impurityName (str): Name of target impurity, only relevant to labelling
            voxelSize (int): List of voxel objects contained in this grain
            iceSize (list): Actual (target) dimensions of ice in micrometers [x,y,z]
            spaceSize (list): Dimensions of modelled space in voxels [x,y,z] 
            targetGrainRad (int): Target grain radius in micrometers
        """
        if self.__class__ is Model:
            raise NotImplementedError("Model should not be instanced directly.")
        self.iceName = iceName
        self.filePath = modelFilePath = f"Models/{iceName}_{impurityName}_{iceSize}_{spaceSize}_{voxelSize}Mics"
        os.makedirs(modelFilePath, exist_ok=True)
        os.chdir(modelFilePath)
        self.impurityName = impurityName
        self.spaceFilePath = "Data/Space.pkl"
        self.interroFilePath = "Data/Interrogator.pkl"
        self.voxelSize = voxelSize
        self.spaceSize = spaceSize
        self.iceSize = iceSize
        self.targetGrainRad = targetGrainRad
        interro = Interrogator(self.spaceFilePath, voxelSize, 
                               spaceSize,
                               iceName, targetGrainRad)
        pickle_Save('Data/Interrogator.pkl', interro)
        
    def load_Space(self):
        """ Loads space object associated with this model """
        return pickle_Load(self.spaceFilePath)

class LAModel(Model):
    """
    Model based on input of LA data to populate structure with impurities.
    Structure is generated using Voronoi growth, with grain and
    boundary objects recorded. An impurity distribution is pushed onto this
    structure by randomly assigning an intensity drawn from a boundary or
    impurity intensity distribution.
    
    Attributes:
        chemicalImage (arr): array containing chemical image
        segmentationImage (arr): array containing image for mask of chemical im
        boundaryDistribution (list): prob. dist. for boundary intensities
        interiorDistribution (list): prob. dist. for interior intensities
    """
    
    def __init__(self, iceName, targetGrainRad, iceSize, chemicalData,
                 impurityName, threshold, voxelSize, segmentationData, 
                 boundariesBool = 1, p = 3):
        # Calculate required values
        targetGrainRadVoxels = int(targetGrainRad/voxelSize)
        targetSpaceSizeVoxels = [int(size / voxelSize) for size in iceSize]
        cellSizeVoxels, actualSpaceSizeVoxels, numCells = get_Cell_Space_Size(targetGrainRadVoxels, targetSpaceSizeVoxels)
        spaceSizeAndBuff = [dim + get_Buffer(cellSizeVoxels) for dim in actualSpaceSizeVoxels]        
        # Init parent class
        super().__init__(iceName, targetGrainRad, voxelSize, impurityName, 
                         iceSize, actualSpaceSizeVoxels)
        self.chemicalImage = chemicalData
        self.segmentationImage = segmentationData
        self.boundaryDistribution, self.interiorDistribution = self.__generate_Probs(chemicalData, segmentationData)
        # Amend volume to account for size of buffer to use and calculate
        # number of cell centers
        numCents = get_Num_Cents(spaceSizeAndBuff, targetGrainRadVoxels)
        # Initalise space object and fill it
        space = Space(cellSizeVoxels, numCells, numCents, voxelSize,
                      boundariesBool = boundariesBool, p = p)
        print("Done making space")
        space.LA_Fill(self.boundaryDistribution, self.interiorDistribution,
                      processing = "series")
        pickle_Save('Data/Space.pkl', space)
        pickle_Save('Model.pkl', self)
        print("Model made")        
        print("--- %s seconds ---" % (time.time() - start_time))

    def __generate_Probs(self, img, seg):
        """
        Generates probability distributions for intensity assignment based on
        arrays of intensities for both the grain interior and boundary
        distributions. Uses sqroot(number of data points) as number of bins
        """
        boundInten, inInten = apply_Mask(img, seg)
        interiorHist = get_Loghist(inInten)
        interiorProb = hist_To_Prob(interiorHist)
        boundaryHist = get_Loghist(boundInten)
        boundaryProb = hist_To_Prob(boundaryHist)
        pickle_Save("Outputs/Interim/Empirical/hists.pkl", [boundaryHist, interiorHist])
        pickle_Save("Outputs/Interim/Empirical/probs.pkl", [boundaryProb, interiorProb])
        return boundaryProb, interiorProb

class Space:
    
    """
    Class to manage the spatial model information. Methods allow generation of
    cells containing voxels, and subsequent classification of these voxels into
    boundary or interior grain while assigning each voxel an intensity value.
    Each cell is a cube of voxels which is small enough to be processed in one
    go. Cells are packed to fill the whole volume. Once generated, the class 
    handles plotting of information in spatial context. Space objects can be 
    directly interacted with, but are mostly interacted with via a facilitating 
    object such as an interrogator or model.
    
    Attributes:
        buffer (int): The number of buffer voxels padded to each dimension to
        ensure entire space is coherent
        cellSize (int): Size of each cell in voxels
        voxelSize (int): Size of voxel in micrometer
        numCells (list):  Total number of cells in the space
        size (int): Size of space in voxels
        loadedCell (cell object): Cell loaded into the space
        numCents (int): Number of voronoi centers in the space
        meanIntensity (float): Mean intensity of all voxels in space
        stdIntensity (float): Standard deviation of voxle intensity
        grainVoxels (int): Number of voxels contained in grains
        boundVoxels (int): Number of voxels contained at grain boundaries
        grainSizeInfo (list): info on grain sizes, as [grain ID, grain size, 
                                                       interior-of-space bool]
        CFA (profile object): Profile object for space's CFA profile
        p (int): space distance metric, usually 3
    """
    
    # cellSize, numCells [x, y, z], totalNumCents, referenceIntensity, referenceSegmentation
    def __init__(self, cellSize, numCells, totalNumCents, voxelSize, 
                 boundariesBool = 1, p = 3):
        # Buffer expects to be at least one
        self.buffer = max(get_Upper_Cell_Size_Voxels() - cellSize, 1)
        self.cellSize = cellSize
        self.voxelSize = voxelSize
        self.numCells = numCells
        self.size = [int(cellSize*numCells[0]), int(cellSize*numCells[1]), int(cellSize*numCells[2])]
        self.loadedCell = None
        # Generate cells, with relevant information
        self.numCents = totalNumCents     
        self.__generate_Cells()
        self.meanIntensity = 0
        self.stdIntensity = float('nan')
        self.grainVoxels = 0
        self.boundVoxels = 0
        self.grainSizeInfo = [] # [grain ID, grain size, interior-of-space bool]
        self.CFA = Profile("CFA", "All", "All", 
                np.zeros(self.size[2], dtype=np.float64).tolist(), 
                self.voxelSize, baseCells = "All", save = "CFA")
        self.p = p # distance metric power
        # Bool to indicate whether to generate boundaries, should be 1 unless
        # testing
        self.boundariesBool = boundariesBool
        
        
    def LA_Fill(self, boundDist, intDist, processing = "series", analyse = 1):
        """
        Function to handle filling the space with a voronoi tesselation. This
        is the main model-building function
        """
        self.boundaryDistribution, self.interiorDistribution = boundDist, intDist
        if analyse:
            # Generate profile metadata to allow profiles to be saved 
            # during generation. Including all one-wide profiles lying on 
            # some face. Profiles are always generated as one-voxel width and 
            # can be later aggregated to multiple voxels to simulate larger spot 
            # sizes. Profile metadata is stored as [xLoc, yLoc]
            faceMeta = []
            for p in range(self.size[1]):
                faceMeta.append([0,p])
                faceMeta.append([self.size[0], p])
            randomMeta = self.__generate_Random_Profiles_Metadata(voxelResolutions = [1,2,3,4], num = 50)
            # Identify which cells each profile runs through, only need to
            # consider bottom plane of cell, cycle through all profiles
            profilesMeta = faceMeta + randomMeta
            profilesBases = [] # base cell through which profiles run
            for c, profileMeta in enumerate(profilesMeta):
                profileBaseCell = []
                # Only need to consider the bottom plane of cells, as all
                # profiles are straight so can infer other cells passed
                # through
                for i in range(self.numCells[0]*self.numCells[1]):
                    lims = self.__cell_Lims(i)
                    if  (lims[0][0] <= profileMeta[0] <= lims[1][0] and
                         lims[0][1] <= profileMeta[1] <= lims[1][1]):
                            profileBaseCell.append(i)
                profilesBases.append(profileBaseCell)
                Profile("LA", profileMeta[0], profileMeta[1], 
                        np.zeros(self.size[2], dtype=np.float64).tolist(), 
                        self.voxelSize, baseCells = profileBaseCell, 
                        save = f"LA_Profile_{c}")
            pickle_Save("Outputs/Interim/Model/ProfilesMeta.pkl", [[profilesMeta], [profilesBases]])
        # Iterate through cells, generating separately, go over cells in each
        # column (iterate over zs) to allow less loading of profiles
        for x in range(self.numCells[0]):
            for y in range(self.numCells[1]):
                # For each new base cell, load profiles in this vertical section
                if analyse:
                    currentProfiles = []
                    profileIDs = []
                    for c, baseCells in enumerate(profilesBases):
                        for baseCell in baseCells:
                            if baseCell == x+y*self.numCells[0]:
                                profileIDs.append(c)
                                currentProfiles.append(pickle_Load(f"Outputs/Interim/Model/LA_Profile_{c}.pkl", printLoad = 0))
                cellGroupIndicies = []
                for z in range(self.numCells[2]):
                    cellIndex =  z*self.numCells[0]*self.numCells[1] + y*self.numCells[0] + x
                    if processing == "series":
                        self.__index_Cell_Fill(cellIndex, currentProfiles)
                    elif processing == "parallel":
                        # FUNCTIONALITY NOT CURRENTLY WORKING                    
                        cellGroupIndicies.append(cellIndex)
                # Save profiles once they are completely populated
                for c, profile in enumerate(currentProfiles):
                    index = profileIDs[c]
                    pickle_Save(f"Outputs/Interim/Model/LA_Profile_{index}.pkl", profile)
        pickle_Save("Outputs/Interim/Model/grainSizes.pkl", self.grainSizeInfo)
        pickle_Save("Outputs/Interim/Model/CFA.pkl", self.CFA)
        return 0

    def __generate_Random_Profiles_Metadata(self, voxelResolutions = [1], num = 1):
        """
        Method to identify which profiles to save. Gets some preset laser 
        ablation profile from a generated space saves profile objects to a
        list self.profiles
        """
        # Initiate profile information, select some random profiles
        profilesMeta = []
        # Create some number of tracks to take average from
        for i in range(num):
            xPos = random.randrange(self.size[0])
            yPos = random.randrange(self.size[1])
            # Create adjacent tracks to allow larger spot sizes to be simulated
            # currently only in x, but statistically x and y are the same
            for offset in voxelResolutions:
                if xPos + max(voxelResolutions) < self.size[0]:
                    profilesMeta.append([xPos + (offset-1), yPos])
                elif xPos - max(voxelResolutions) > 0:
                    profilesMeta.append([xPos - (offset-1), yPos])
                else:
                    # in this case, space is very small
                    print("Space too small to suppot larger spot sizes")
                    profilesMeta.append([xPos, yPos])
                    break
        return profilesMeta        

    def load_Cell(self, cellIdentifier):
        """
        Loads  specified cell from disk
        """ 
        if cellIdentifier > np.prod(self.numCells):
            print("An attempt to load a cell outside the space was made")
            return 0
        elif self.loadedCell != None and cellIdentifier == self.loadedCell.identifier:
            return self.loadedCell
        else:
            location = os.getcwd()
            fp = str(location) + '\\Cells\\cell' + str(cellIdentifier) + '.pkl'
            self.loadedCell = pickle_Load(fp)
            return self.loadedCell

    # --- Generation functions
    
    def __generate_Cells(self):
        """
        Generates a list of random grain centers and corresponding identifiers 
        for whole space to be used as voronoi centers. Creates sublists of
        these centers for each cell. Can generate at least 10^8 centers 
        """ 
        #Creates cell objects and saves them
        centsIdents = self.__generate_Centers()
        for i in range(np.prod(self.numCells)):
            # Create cell and save it
            Cell(i, centsIdents[i])
        return 0

    def __generate_Centers(self):
        """
        Generates a list of random grain centers and corresponding identifiers 
        for whole space to be used as voronoi centers. Creates sublists of
        these centers for each cell. Can generate at least 10^8 centers 
        """  
        c = self.buffer
        centers = np.random.randint(
            (-c,-c,-c), 
            (self.size[0] + c, self.size[1] + c, self.size[2] + c), 
            (self.numCents, 3))       
        # Attach an identifying integer to each point in `points`
        ids = np.arange(self.numCents)        
        centersWithids = np.hstack((centers, ids.reshape(-1, 1)))
        # Sort the points by their z, then y, then x coordinate
        centersWithids = centersWithids[np.lexsort((centersWithids[:, 2], centersWithids[:, 1], centersWithids[:, 0]))]        
        # Select points that fall in the cell space, with some buffer
        cellsGrainCenters = []
        for k in range(self.numCells[2]):
            for j in range(self.numCells[1]):
                for i in range(self.numCells[0]):
                    cellsGrainCenters.append(centersWithids[
                    np.logical_and.reduce((
                        centersWithids[:, 0] > self.cellSize*i-c, centersWithids[:, 0] < self.cellSize*(i+1) + c,
                        centersWithids[:, 1] > self.cellSize*j-c, centersWithids[:, 1] < self.cellSize*(j+1) + c,
                        centersWithids[:, 2] > self.cellSize*k-c, centersWithids[:, 2] < self.cellSize*(k+1) + c
                    ))])
        # TO DO - ADD SOME CHECK TO ENSURE THERE ARE NO EMPTY CELLS, may be
        # relevant for large spaces with few grains
        return cellsGrainCenters
    
    
    """ Cell population functions """
    
    
    def __index_Cell_Fill(self, index, profiles):
        """ Fill cell of given index with tessellation and impurity imprint """
        cell = self.load_Cell(index)
        # Feed points into voronoi grid generator, get 3D image with unique regions
        # represented with a different value
        lowLimits, highLimits = self.__cell_Lims()        
        # Apply buffer
        highLimits = np.add(highLimits, self.buffer)
        lowLimits = np.subtract(lowLimits, self.buffer)        
        cellAssigmentsBuffer = voronoi_3D([(cent[0], cent[1], cent[2]) for cent in cell.centsIdents],
                                         [(cent[3]) for cent in cell.centsIdents], 
                                         lowLimits, highLimits, metric='minkowski', p=self.p)  
        # Function to identify boundaries in voronoi grid and remove buffer and
        # to put data into grain and boundary features for each cell       
        self.__make_Features(cellAssigmentsBuffer)
        if profiles != None:
            self.__update_Profiles(cell, profiles)
        cell.saveCell()
        return 0
    
    def __update_Profiles(self, cell, profiles):
        """ Update intensity values for some profile """
        # go over every voxel and update relevant profiles
        for feature in itertools.chain(cell.boundaries, cell.grains):
            for voxel in feature.voxels:
                for profile in profiles:
                    if self.__space_Coord(voxel.coord)[0] == profile.xPos and self.__space_Coord(voxel.coord)[1] == profile.yPos:
                        # Update profile
                        profile.intensity[self.__space_Coord(voxel.coord)[2]] += voxel.intensity
                        profile.voxelType[self.__space_Coord(voxel.coord)[2]] = feature.identifier
        return 0
   
    def __make_Features(self, img, printInfo = 0):
        """Method to take np array of region, with buffer, storing unique 
        values for each grain interior and grain boundary and generate grid. 
        This method manages translation from np array coord system to 3d space 
        """
        # Iterate over every element in array storing unqiue values for each
        # voronoi region. Assign boundary value if point is found to be at 
        # boundary. Indicies refer to the original voroni space, which includes 
        # a buffer which is later removed
        self.stdTemp = 0 # temp variable for calculating intensity standard dev
        lowerPost = self.buffer
        upperPost = self.cellSize + self.buffer
        # Change this section to change boundary creation behaviour. Currently
        # boundaries are double-thick (as the original image is not updated
        # as new boundaries are added) creating tessellations well-suited for
        # representing 40 micron spot sizes
        for i in range(lowerPost, upperPost):
            for j in range(lowerPost, upperPost):
                for k in range(lowerPost, upperPost):
                    # Create set to store which grains are nearby each cell
                    grainsPresent = set([img[i,j,k]])
                    # Compare the current point with its surrounding points
                    for ii in range(i-1, i+2):
                        for jj in range(j-1, j+2):
                            for kk in range(k-1, k+2):
                                # Skip the current point
                                if ii == i and jj == j and kk == k:
                                    continue
                                # Skip diagonally adjacent cells
                                if abs(ii-i) + abs(jj-j) + abs(kk-k) != 1:
                                    continue
                                if img[i,j,k] != img[ii,jj,kk]:
                                    grainsPresent.add(img[ii,jj,kk])
                     # Check if point is boundary or grain interior
                    if len(grainsPresent) == 1 or self.boundariesBool == 0:
                        # Point is interior of grain, add to appropriate grain
                        grainID = next(iter(grainsPresent))
                        self.__assign_Grain_Voxel([j-self.buffer,i-self.buffer,k-self.buffer],
                                                  grainID, self.loadedCell)                       
                    else:
                        # Point is boundary, add to appropriate boundary                        
                        self.__assign_Boundary_Voxel([j-self.buffer,i-self.buffer,k-self.buffer], grainsPresent, self.loadedCell)
                        
        # Calculate standard deviation for intensity values
        self.stdIntensity = (self.stdTemp / (self.grainVoxels + self.boundVoxels - 1)) ** 0.5
        if printInfo == 1:
            print(f"Grains considered for voronoi for cell at {self.__cell_Lims()}:")
            print(self.loadedCell.centsIdents)
            print(f"Grains with influence in cell (total {len(self.loadedCell.grains)}): ")
            for grain in self.loadedCell.grains:
                print(f"Grain ID: {grain.identifier}")
        return 0
    
    def __update_intensity_mean_std(self, intensityVal):
        """
        Method to update the mean and deviation for total intensity in space.
        Ensures new entries are appropriately weighted.
        """
        count = self.grainVoxels + self.boundVoxels + 1
        delta = intensityVal - self.meanIntensity  # Calculate delta for standard deviation update
        self.meanIntensity += delta / count  # Update mean iteratively
        self.stdTemp += delta * (intensityVal - self.meanIntensity)  # Update intermediate value for standard deviation
        return 0
    
    def __assign_Grain_Voxel(self, coord, grainID, cell):
        """Assign a voxel to a grain feature"""
        # Randomly assign intensity from boundary intensity distriubtion
        voxInt = np.random.choice(self.interiorDistribution[1], p=self.interiorDistribution[0], size=1)[0]
        self.CFA.intensity[self.__space_Coord(coord)[2]] += voxInt
        self.__update_intensity_mean_std(voxInt)
        # Indicate whether voxel is on space edge, to be recorded for grain
        if 0 in coord or any(c1-1 == c2 for c1, c2 in zip(self.size, self.__space_Coord(coord))):
            internal = 0
        else: internal = 1
        # If grain present add to existing object
        if grainID in cell.grainIDs:
            # append to feature
            ind = cell.grainIDs.index(grainID)
            cell.grains[ind].addVoxel(Voxel(coord, voxInt), internal)
            # Update grain info for later analysis
        # else grain not present - create new grain object
        else:
            # create feature
            grainInfo = next((sublist for sublist in cell.centsIdents if sublist[3] == grainID), [])
            cell.grains.append(Grain(grainInfo, Voxel(coord, voxInt), internal))
            cell.grainIDs.append(grainID)
        # Update trackers for later analysis
        self.__update_Grain_Size_Info(grainID, internal)
        self.grainVoxels += 1
        return 0
    
    def __assign_Boundary_Voxel(self, coord, grainsSeparates, cell):
        """Assign a voxel to a boundary feature"""
        # Randomly assign intensity from boundary intensity distriubtion
        voxInt = np.random.choice(self.boundaryDistribution[1], p=self.boundaryDistribution[0], size=1)[0]
        self.CFA.intensity[self.__space_Coord(coord)[2]] += voxInt
        self.__update_intensity_mean_std(voxInt)
        # if boundary present - add voxel to it
        if grainsSeparates in cell.boundariesSeparates:
            # append to feature
            ind = cell.boundariesSeparates.index(grainsSeparates)
            cell.boundaries[ind].addVoxel(Voxel(coord, voxInt))
        # else boundary not present - create new boundary object
        else:
            # create feature
            cell.boundaries.append(Boundary(-1, grainsSeparates, Voxel(coord, voxInt)))
            cell.boundariesSeparates.append(grainsSeparates)
        self.boundVoxels += 1
        return 0
    
    def __update_Grain_Size_Info(self, ID, edgeBool):
        """ Updates grain size info stored in space object"""
        # Assume grain ID is not found to start with
        found = False
        # Iterate through grain size infos attempting to match ID value
        for triplet in self.grainSizeInfo:
            if triplet[0] == ID:
                # If ID value is found, increment y by 1 and multiply z
                triplet[1] += 1
                triplet[2] *= edgeBool
                found = True
                break
        # If x value is not found, create a new triplet entry
        if not found:
            self.grainSizeInfo.append([ID, 1, edgeBool])
        return 0
            
    def __index_Cell_Pool(self, index):
        """ Unused parallelisation method"""
        cell = self.load_Cell(index)
        # Pool planes and add to overall CFA list
        for feature in itertools.chain(cell.boundaries, cell.grains):
            for voxel in feature.voxelss:
                self.unsmoothedCFA[self.__space_Coord(voxel.coord)[2]] += voxel.intensity
        return 0
    
    
    """ Coordinate system methods """
    
    
    def __cell_Lims(self, index = None):
        """Method to get upper and lower limit coordinates of space"""
        cellIndex = self.__cell_Coords(index)
        lowLims = [cellIndex[0]*self.cellSize, cellIndex[1]*self.cellSize, cellIndex[2]*self.cellSize]
        highLims = [lowLims[0]+self.cellSize-1, lowLims[1]+self.cellSize-1, lowLims[2]+self.cellSize-1]
        return lowLims, highLims
    
    def __cell_Coords(self, cellIndex = None):
        """ get cell origin coordinates in space """
        if cellIndex == None:
            cellIndex = self.loadedCell.identifier
        z = cellIndex // (self.numCells[0] * self.numCells[1])
        cellIndex -= z * self.numCells[0] * self.numCells[1]
        y = cellIndex // self.numCells[0]
        x = cellIndex % self.numCells[0]        
        return [x,y,z]
    
    def coord_To_Index(self, coords):
        """Convert a coordinate to an index in an array"""
        x, y, z = coords
        return z*self.numCells[0]*self.numCells[1] + y*self.numCells[0] + x
    
    def __cells_With_Coords(self, coordsList):
        """ Returns list of cell indicies that contain specified coordinates"""
        cellIndicies = set([])
        for coord in coordsList:
            cellIndicies.add(self.coord_To_Index([coord[0]//self.cellSize, 
                                                coord[1]//self.cellSize, 
                                                coord[2]//self.cellSize]))
        return list(cellIndicies)
    
    
    """ Plotting methods """
    
    
    def __build_Voxel_List(self, cellIndicies, coords, mode):
        """
        Method to build a list of coords and colours for voxel plot from a list
        of cell indicies and coords to plot. It iterates over all voxels in 
        relevant cells and adds them to a list if their coord appears in the 
        reference list
        """
        points = []
        colours = []
        for cellIndex in cellIndicies:
            cell = self.load_Cell(cellIndex)
            for feature in itertools.chain(cell.boundaries, cell.grains):
                for voxel in feature.voxels:
                    # Must change voxel coord to be in full-space context
                    spaceCoord = self.__space_Coord(voxel.coord)
                    if spaceCoord in coords:
                        # Remove coord from check, to speed up future searches
                        coords.remove(spaceCoord)
                        points.append(spaceCoord)
                        # Append either grain ID OR intensity, as requested
                        colours.append(self.__return_Colour(feature, voxel, mode))
        if mode == "intensity":
            # Normalise intensities
            norm = plt.Normalize()
            colours = plt.cm.inferno(norm(colours))
        return points, colours
    
    def plot_Cell(self, cellIndex, bounds = "surface", mode = "grains",
                  ax = None, context = 1):
        """
        Plots the entire contents of a specified cell
                
        arguments:
            cellIndex (int): ID number of target cell
            bounds (str): Either "surface" or "volume" - specifying which
            voxels to plot
            mode (str): Either "grains" or "intensity" to specify voxel colours
            ax: axis object on which to make plot
            context (bool): if yes, plots contextual information about cell
        """
        self.load_Cell(cellIndex)
        lowLims, highLims = self.__cell_Lims()
        coords = generate_Coordinates(lowLims, highLims, bounds)
        points, colours = self.__build_Voxel_List([cellIndex], coords, mode)
        ax = self.__plot_Voxels(points, colours, ax = ax, labels = context)
        if context == 1:
            #ax.text2D(0.8,1,"Cell ID: " + str(self.loadedCell.identifier), transform=ax.transAxes)
            #ax.text2D(0.8,0.9,"Cell Coordinate: " + str(self.__cell_Coords()), transform=ax.transAxes)
            ax.text(-10, -10, -10, str(self.__cell_Lims()[0]), "x", fontsize = 28)
        return ax
    
    """UNUSED & UNTESTED
    def plot_Subset(self, lowLims, highLims, mode = "grains", rep = "3D", bounds = "volume"):
        # Plot a subset of the space in either 2D or 3D
        # Generate relevant coords list
        spaceVerts = cuboidVerts(lowLims, highLims, self.cellSize)
        cellIndicies = self.__cells_With_Coords(spaceVerts)
        # Generate full coords list
        coords = generate_Coordinates(lowLims, highLims, bounds)
        if rep == "3D":
            # Call function to generate list of coordinates and colours
            points, colours = self.__build_Voxel_List(cellIndicies, coords, mode)
            # Call plot function
            self.__plot_Voxels(points, colours)
            # Call function to plot each cell outlined in some colour
            for ind in cellIndicies:
                self.load_Cell(ind)
                low, high = self.__cell_Lims()
            return 0
    """
    
    def __space_Coord(self, coord):
        """
        Function to return coordinate in full space context
        """
        cellOrigin = self.__cell_Lims()[0]
        return [coord[0] + cellOrigin[0], coord[1] + cellOrigin[1], coord[2] + cellOrigin[2]]
    
    def __return_Colour(self, feature, voxel, mode):
        """
        Returns colours for plots depending on voxel's parameters
        """
        if mode == "grains":
            if isinstance(feature, Boundary):
                return("#33333355")
            if isinstance(feature, Grain):
                coloursList = get_Colours()
                return(coloursList[feature.identifier%len(coloursList)])
        elif mode == "intensity":
            return(voxel.intensity)
    
    def __plot_Voxels(self, positions, colors, ax = None, save = 0, labels = 0):
        """
        Method to plot voxels at specific (x,y,z) coordinates with a specific 
        colour. Arguments are coords and colours as separate lists
        """
        if ax == None:
            plt.figure(figsize=(10, 10), dpi=40)
            ax = plt.axes(projection='3d')
            ax.set_axis_off()
        # Create cubes
        pc, lowLims, upLims = create_Cubes_And_Lims(positions, colors = colors,
                                                    edgecolor="#00000044")
        # Add cubes to axs
        ax.add_collection3d(pc)
        # Create bounding box to force equal aspect ratio and square voxels
        # Compute the center of the space
        center = [(upper + lower) / 2 for upper, lower in zip(upLims, lowLims)]
        # Compute the maximum range based on the space dimensions
        max_range = max([upper - lower for upper, lower in zip(upLims, lowLims)])
        zoom_factor = 2  # Adjust the zoom level as desired
        max_range *= zoom_factor
        # Set the limits for the plot
        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
        # Create the cubic bounding box
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + center[0]
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + center[1]
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + center[2]
        
        # Plot the bounding box vertices
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        
        # Show the plot
        ax.set_zlabel("z")
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        
        if labels == 1:
            x = max(coord[0] for coord in positions)
            y = max(coord[1] for coord in positions)
            z = max(coord[2] for coord in positions)

            """
            # Add space limits
            ax.plot([0, limits[0]], [limits[1],limits[1]],
                    zs=[limits[2],limits[2]], color = 'black')
            ax.plot([limits[0], limits[0]], [0,limits[1]],
                    zs=[limits[2],limits[2]], color = 'black')
            ax.plot([limits[0], limits[0]], [limits[1],limits[1]],
                    zs=[0,limits[2]], color = 'black')
            """
            # Add axes arrows
            xLine = ax.plot([0, 0.1], [0, 0], [0, 0], linewidth=0.1,
                            label='x-axis', color='blue')
            yLine = ax.plot([x+1, x+1], [0, 0.1], [0, 0], linewidth=0.1,
                            label='y-axis', color='blue')
            zLine = ax.plot([0, 0], [0, 0], [0, 0.1], linewidth=0.1,
                            label='z-axis', color='blue')
            
            def add_arrow(line, label, xyz, direction):
                x, y, z = xyz
                ax.annotate(label, xy=(x, y), xytext=(x + direction[0], y + direction[1]),
                            arrowprops=dict(facecolor='blue', shrink = 10, width=10, headwidth=8),
                            ha='center', va='center')
                ax.quiver(x, y, z, direction[0], direction[1], direction[2], color='blue', length=10)
            
            # Add arrows with annotation
            add_arrow(xLine[0], 'X', (x, 0, 0), (1, 0, 0))
            add_arrow(yLine[0], 'Y', (x, y, 0), (0, 1, 0))
            add_arrow(zLine[0], 'Z', (0, 0, z), (0, 0, 1))
            
            # Adding labels to the ends of the lines
            ax.text(x+7, 1, 4, "x", color='blue', fontsize=40, ha='left', va='center')
            ax.text(x+3, y+10, 2, "y", color='blue', fontsize=40, ha='right', va='bottom')
            ax.text(1, 2, z+8, "z", color='blue', fontsize=40, ha='right', va='bottom')

        return ax
    
    def plot_Feature(self, cellIndex, featID = None, featInd = None, ax = None,
                     mode = "grains", featureType = "grain", labels = 0):
        """
        Method to plot a feature contained in some cell. Creates a 3d plot 
        based on specified parameters
        arguments:
            cellIndex (int): ID number of target cell that feature is in
            featID (int): ID number of feature
            featInd (int): Index of feature within cell's features list
            mode (str): Either 'grains' to plot in grain-representation or 
            'intensity' to plot in intensity-representation
            featureType (str): Either 'grain' to select grain features or 
            'boundary to select boundary features'
        """
        if featID == None and featInd == None:
            print("Give either feature ID number or index for feature in list")
            return 0
        cell = self.load_Cell(cellIndex)
        
        if featureType == "grain":
            featList = cell.grains
        elif featureType == "boundary":
            featList = cell.boundaries
        else:
            print("featureType should be either 'grain' or 'boundary'")
        if len(featList) == 0:
            print("No features in list")
            return 0
        if featInd != None:
            feature = featList[featInd]
        if featID !=None:
            for feat in featList:
                if feat.identifier == featID:
                    feature = feat
                    break
        points = []
        colours = []
        if mode == "grains":
            for voxel in feature.voxels:
                points.append(voxel.coord)
                colours.append(self.__return_Colour(feature, voxel, mode))
            self.__plot_Voxels(points, colours, ax = ax, labels = labels)
        elif mode == "intensity":
            for voxel in feature.voxels:
                points.append(voxel.coord)
                colours.append(voxel.intensity)
            norm = plt.Normalize()
            colours = plt.cm.afmhot(norm(colours)) 
            self.__plot_Voxels(points, colours, ax = ax, labels = labels)
        return plt
    
    
class Cell:
    """
    Class for cells that make up full space. All cells are cuboidal an have the
    same size. centsIdents holds information on the coordinates of the seed 
    points used to generate this cell's voronoi tesselation. Actual grains are
    a subset of these identities (as much of the Voronoi tesselation is 
    discarded to allow continuity between calls). Uses coordinate grid where 
    bottom left voxel in cell is (0,0,0).

    Attributes:
        identifier (int): Cell's unique ID 
        centersInfo (list): List of coordinates and IDs of grains in cell
        boundaries (list): List of boundary objects in cell
        boundariesSeparates (list): Boundary identifiers
        grains (list): List of grain objects in cell
        grainIDs (list): Grain identifiers
    """

    def __init__(self, identifier, centersInfo):
        """
        Args:
            identifier (int): Cell's unique identifier
            centersInfo (list): List of coordinates and IDs of grains in cell
        """
        self.identifier = identifier
        self.centsIdents = centersInfo
        self.boundaries = []
        self.boundariesSeparates = []
        self.grains = []
        self.grainIDs = []
        self.saveCell()
        
    def saveCell(self):
        """ Saves the cell object to a pickle file """
        location = os.getcwd()
        fp = str(location) + "\\Cells\\cell" + str(self.identifier) + '.pkl'
        pickle_Save(fp, self)
        return 0
    
    def get_Grain(self, index):
        """
        Retrieves a grain based on index passed to grains attribute
        Args:
            index (int): Index of the grain to retrieve
        Returns:
            Grain: The grain object located at the index passed to function
        """
        for grain in self.grains:
            if grain.identifier == index:
                break
        return grain

class Boundary:
    """
    Attributes:
        identifier (int): Unique identifier for the boundary
        separates (list): List of grain object IDs which this bound separates
        voxels (list): List of voxel objects contained in this boundary
    """
    def __init__(self, identifier, separatedBoundaries, coord):
        self.identifier = identifier
        self.separates = separatedBoundaries
        self.voxels = [coord]
    def addVoxel(self, voxel):
        """Adds a voxel obj to the boundary's list of voxels"""
        self.voxels.append(voxel)
    def removeVoxel(self, voxel):
        """Removes a voxel obj from the boundary's list of voxels"""
        self.voxels.remove(voxel)

class Grain:
    """    
    Attributes:
        identifier (int): Unique identifier for the grain
        center (list): Coordinates of the grain's inital voxel obj
        internal (int): Flag showing if grain is internal to space (1 yes, 0 no)
        voxels (list): List of voxel objects contained in this grain
    """
    def __init__(self, info, voxel, internal):
        """
        Args:
            info (list): Information about the grain including center coordinates and identifier
            voxel (object): Voxel object at the seed point of the grain
            internal (int): Flag showing if inital voxel
        """
        self.identifier = info[3]
        self.center = [info[0], info[1], info[2]]
        self.internal = internal
        self.voxels = [voxel]

    def addVoxel(self, voxel, internal):
        """Adds a voxel obj to the grain's list of voxels
        Also updates whether grain remanes internal to the space"""
        self.voxels.append(voxel)
        if internal == 0:
            self.internal = 0

    def removeVoxel(self, voxel):
        """Removes a voxel obj from the grain's list of voxels"""
        self.voxels.remove(voxel)
        
class Voxel:
    """ Voxels coordinate and intensity info.
    The coordinate is a within-cell coordinate, so must be translated into
    a full-space coordinate if plotting or analysing more than one cell """
    def __init__(self, coord, intensity):
        self.coord = coord
        self.intensity = intensity

class Profile:
    """    
    A class to store 1D intensity data taken from a space object, and 
    associated metadata
    Attributes:
        profileType (int): CFA or LA
        xPos, yPos (list): in form [lowerLim, upperLim]
        spotSize (int): Laser spot size the profile represents
        intensity (list): Intensity value of each voxel
        voxelType (list): ID number of the feature the voxel's belong to
        baseCells (list): List of cells the profile runs through
    Methods:
        plot_Profile: Plots the profile's intensity profile against depth
    """
    def __init__(self, profileType, xPos, yPos, lineData, spotSize, 
                 baseCells = None, save = 0):
        self.profileType = profileType
        self.xPos = xPos # in form [lowerLim, upperLim]
        self.yPos = yPos # in form [lowerLim, upperLim]
        self.spotSize = spotSize
        self.intensity = lineData
        self.voxelType = [None]*len(lineData)
        self.baseCells = baseCells
        if save != 0:
            pickle_Save(f"Outputs/Interim/Model/{save}.pkl", self)
            
    def plot_Profile(self, colour = "Blue", ax = None, 
                     label = None, smoothing = None):
        """        
        Plots profile intensity vs depth variation
        """
        if isinstance(colour, int):
            colour = get_Colours()[colour]
        ytoPlot = self.intensity
        xtoPlot = [i * self.spotSize for i in range(len(ytoPlot))]
        if smoothing != None:
            ytoPlot = gaussian_filter1d(ytoPlot, smoothing, mode='reflect')
        ax.set_xlim(0, max(xtoPlot))
        graph_Profile([ytoPlot, xtoPlot],
                    label =  label,
                    orientation = "Horizontal",
                    colour = colour,
                    ax = ax)
        return 0


class Interrogator:
    """    
    A class to store data on a space to be used for analysis. Designed to be
    less data intensitve to interrogate than the space object itself. In this
    implementation only profiles from one face are saved and analysed. Can also
    be amended to directly operate on cells to extract info.
    
    Attributes:
        iceName (str): Name of modelled ice
        spaceFilePath (str): File path to associated space
        voxelDimension (int): Laser spot size in microns
        spaceSize (list): Size of space in voxels in form (x,y,z)
        targetGrainRad (int): Targetted effective grain radius in microns
        ratioBoundInt (int): Expected ratio of GB to GI voxels
        volumeIntensity (list): [mean voxel intensity, std vox intensity]
        numInteriorVoxels (int): Number of voxels in grain interiors
        numBoundaryVoxels (int): Number of voxels at grain boundaries
        cfaRaw (Profile object): Profile object for CFA intensity
        
    """
    def __init__(self, spaceLoc, voxelDimension, spaceSize, iceName, targetGrainRad):
        # Filepath to a saved space object
        self.iceName = iceName
        self.spaceFilePath = spaceLoc
        self.voxelDimension = voxelDimension
        self.spaceSize = spaceSize
        self.targetGrainRad = targetGrainRad
        self.ratioBoundInt = ratio_Boundary(targetGrainRad)
    
    def extract_Info(self):
        """        
        Loads space object and extracts information
        """
        space = self.load_Space()
        self.volumeIntensity = [space.meanIntensity, space.stdIntensity]
        self.numInteriorVoxels = space.grainVoxels
        self.numBoundaryVoxels = space.boundVoxels
        self.cfaRaw = pickle_Load("Outputs/Interim/Model/CFA.pkl")
        self.__process_CFA()
        self.__get_Face_From_Profiles()
        #self.generate_MAD()
        return 0


    def generate_MAD(self, toInclude = None):
        """        
        Generates MAD objects for some profiles
        """
        self.CFAMAD = np.mean(np.abs(self.smoothedCFASig - self.volumeIntensity[0]))*100
        
        # Get face array to generate MADS from
        array = self.faceIntensityArray
        # Set parameters for LA mads to calculate
        minNumProfiles = 1
        spotSizes = [1,7]
        smoothingsForSpots = [[1, 7, self.cfaSmoothing],
                      [1, 12/7, self.cfaSmoothing/7]]
        self.MADspotSizes = spotSizes
        self.smoothingsForMAD = smoothingsForSpots
        totalGroups = 55
        #smoothings are in micrometers
        self.MADObjects = []
        # To manage different spot sizes - downsample 2d array to appropirate
        # resolution and process in the same way
        for spotSize, smoothings in zip(spotSizes, smoothingsForSpots):
            # Downsample array by averaging into blocks
            downsampledShape = (array.shape[0] // spotSize, array.shape[1] // spotSize)
            blocks = array[:downsampledShape[0] * spotSize, :downsampledShape[1] * spotSize].reshape(downsampledShape[0], spotSize, downsampledShape[1], spotSize)
            downsampledArr = blocks.mean(axis=(1, 3))
            maxNumProfiles = downsampledArr.shape[1]
            for smoothing in smoothings:
                # Create list of numbers of profiles to work with. Space number of profiles
                # logarithmically apart, as larger separations are less interesting
                numProfiles = np.logspace(np.log10(minNumProfiles), np.log10(maxNumProfiles), num=totalGroups, endpoint=True)
                if not np.isin(toInclude, numProfiles):
                    # Append the value
                    numProfiles = np.append(numProfiles, toInclude)
                if toInclude not in numProfiles:
                    numProfiles.append(toInclude)
                numProfiles = sorted(set(np.round(numProfiles).astype(int)))
                # Iterate through numbers of profiles to combine
                for size in numProfiles:
                    # Ensure num_combinations is not greater than the total number of combinations
                    numCombis = min(200, factorial(downsampledArr.shape[1]) // (factorial(size) * factorial(downsampledArr.shape[1] - size)))
                    # Generate combinations on-the-fly and keep only the desired number
                    combinationsList = set()
                    while len(combinationsList) < numCombis:
                        newCombination = tuple(sorted(random.sample(range(downsampledArr.shape[1]), size)))
                        combinationsList.add(newCombination)
                    # Iterate over profile combinations of the current size, and store
                    # their MAD values
                    for combination in combinationsList:
                        # Calculate combined signal
                        combinedSignal = np.mean(downsampledArr[:, combination], axis=1)
                        # Smooth signal
                        combinedSmoothed = gaussian_filter1d(combinedSignal, smoothing)
                        # Calculate MAD 
                        MADUnnorm = np.mean(np.abs(combinedSmoothed - self.volumeIntensity[0]))
                        MAD = (MADUnnorm * 100)/self.volumeIntensity[0]
                        # Update list storing MADs for each number of profile combinations
                        self.MADObjects.append(MADClass(spotSize, smoothing, MAD, combination))
        return 0


    def __get_Face_From_Profiles(self):
        """
        Extracts face array data from profiles saved during volume growth
        """
        profiles = self.load_Profiles(duringGrowth = 1)
        # Prep empty face array of correct size
        faceIntensityArray = np.zeros((self.spaceSize[2], self.spaceSize[1]), dtype=np.float64)
        faceGrainArray = np.zeros((self.spaceSize[2], self.spaceSize[1]), dtype=np.float64)
        # Cycle through all profiles and select ones that lie on face and have
        # a spot size of the space resolution
        faceProfiles = []
        for profile in profiles:
            if profile.xPos == 0:
                # Populate the face array with relevant data
                faceIntensityArray[:, profile.yPos] = profile.intensity         
                faceProfiles.append(profile)
                faceGrainArray[:, profile.yPos] = profile.voxelType
        self.faceIntensityArray = faceIntensityArray
        self.faceGrainArray = faceGrainArray
        self.faceProfiles = faceProfiles
        return faceIntensityArray, faceProfiles, faceGrainArray
        
    def __process_CFA(self):
        # Get and smooth cfa
        cfaSmoothing = int(10000 / self.voxelDimension) # 1cm smoothing
        cfaSmoothed = gaussian_filter1d(self.cfaRaw.intensity, cfaSmoothing)
        cfaPerVolume = cfaSmoothed / (self.spaceSize[0]*self.spaceSize[1])
        self.cfaSmoothing = cfaSmoothing
        self.smoothedCFASig = cfaPerVolume
        return 0
    
    def load_Profiles(self, duringGrowth = 1):
        """
        Loads saved profiles
        """
        if duringGrowth == 0:
            # UNUSED CASE
            return pickle_Load("Outputs/Interim/Model/profiles.pkl")
        if duringGrowth == 1:
            folder = 'Outputs/Interim/Model/'
            contains = 'LA_Profile'
            # Get a list of la profile file paths
            paths = [os.path.join(folder, fileName) for fileName in os.listdir(folder) if contains in fileName]
            # Load all profiles into a list
            profiles = []
            for path in paths:
                profiles.append(pickle_Load(path, printLoad = 0))
            return profiles
    
    # --- Data Plotting Methods --- #    
    
    def plot_Grain_Sizes(self, ax = "new", typeToPlot = "all", normaliseSize = 0, save = 0):
        """
        Plot information on grain sizes
        Parameters:
            typeToPlot: "all" for all grains, "interior" for only interior
            grains, "exterior" for only exterior grains
        Returns:
            ax: the axis created for the plots
        """
        zippedData = pickle_Load("Outputs/Interim/Model/grainSizes.pkl")
        ids, sizes, interiorBools = zip(*zippedData) # zipped format: [grainIdentifiers, grainSizes, grainInteriorBools]
        if ax == "new":
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(figHeight, figWidth)
        # Select which grains to plot
        if typeToPlot == "interior":
            # Select only grains marked as interior
            toPlot = [x for x, bi in zip(sizes, interiorBools) if bi == 1]
            title = "Grain sizes for interior grains"
            numberText = f'Number of grains in subset: {len(toPlot)}'
            volumeText = 'Mean subset grain volume: '
            radiusText = 'Mean subset grain radius: '
        elif typeToPlot == "exterior":
            # Select only grains marked as exterior
            toPlot = [x for x, bi in zip(sizes, interiorBools) if bi == 0]        
            title = "Grain sizes for exterior grains"
            numberText = f'Number of grains in subset: {len(toPlot)}'
            volumeText = 'Mean subset grain volume: '
            radiusText = 'Mean subset grain radius: = '
        else:
            # Take all grains
            toPlot = sizes
            title = "Grain sizes for all grains"
            numberText = f'Total number of grains N = {len(toPlot)}'
            volumeText = 'Mean grain volume v\u0304 = '
            radiusText = 'Mean grain radius r\u0304 = '
            
        if len(toPlot) == 0:
            print("No data in histogram, not plotting")
            return 0
        # If requested, normalise the size of the grains
        meanVolNum = np.mean(toPlot)
        meanRadNum = volume_To_Radius(meanVolNum)
        voxVolume = self.voxelDimension*self.voxelDimension*self.voxelDimension
        meanRad = meanRadNum*self.voxelDimension
        if normaliseSize == 1:
            toPlot = [x / meanVolNum for x in toPlot]
            plot_Expected_Grain_Dist(ax)
        # Make plot
        plot_Hist(toPlot, ax)
        # Add information on plot
        # Add title, x label, y label, and text information
        axis_Visuals(ax, xLabel = 0, yLabel = 0, legendTitle = title, log = 0, scale = None) 
        ax.set_ylabel('Relative frequency', fontsize=28)
        ax.set_xlabel('Normalised grain volume y', fontsize=28)
        # Add text information
        ax.text(0.95, 0.75, numberText, fontsize=26, color='k', horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)        
        ax.text(0.95, 0.7, volumeText + f'{sci_notation(meanVolNum*voxVolume, precision=None)} \u03BCm', fontsize=26, color='k', horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
        ax.text(0.95, 0.65, radiusText + f'{meanRad:.0f} \u03BCm', fontsize=26, color='k', horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
        plt.title(f'Modelled {self.iceName} profiles', fontsize = 36)
        if save == 1:
            ax.get_figure()
            check_Create_Filepath('Outputs\\Plots\\GrainSize\\')
            plt.savefig(f'Outputs\\Plots\\GrainSize\{typeToPlot}.png', bbox_inches='tight')
        return ax

    def load_Space(self):
        return pickle_Load(self.spaceFilePath)

         
class MADClass:
    """ Simple class to store instances of MAD for profiles"""
    def __init__(self, spotSize, smoothing, MAD, profileIDs):
        self.spotSize = spotSize
        self.smoothing = smoothing
        self.MAD = MAD
        self.profileIDs = profileIDs

# --- Voronoi generation function --- #        

def voronoi_3D(points, pointsIDs, lowLims, highLims, **kwds):
    """
    Function to generate a 3D voronoi tessellation from a list of coordinates
    to serve as the region centers, unique IDs for these regions, and the
    limits of the volume. If points are randomly distributed in the 3D volume, 
    the result will be a Poission voronoi tessellation.

    Parameters:
        points (array): 3D points representing Voronoi centers
        pointsIDs (array): unique identifiers for Voronoi centers
        lowLims (list): x,y,z coordinates for lower bounds of space
        highLims (list): x,y,z coordinates for upper bounds of space
        **kwds: Keyword arguments to pass to the cdist function, currently 
        information on the chosen metric

    Returns:
        positionLabel: A 3D numpy array with each element labelled with the ID
        of its closest point
    """
    # Generate grid with specified limits
    X, Y, Z = np.meshgrid(
        np.arange(lowLims[0], highLims[0]+1, 1),
        np.arange(lowLims[1], highLims[1]+1, 1),
        np.arange(lowLims[2], highLims[2]+1, 1),
        indexing='xy'
    )
    z = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    # Determine, for each point in the grid, the closest point in the set of 
    # input points
    # TO DO - ADD SOME CHECK HERE (OR BEFORE) TO SEE IF EACH CELL HAS A CENTER
    # CURRENTLY IF A CELL DOES NOT HAVE ANY CENTERS THE CODE CRASHES HERE
    dists = cdist(points, z, **kwds)
    # Find the index of the closest point (region center) for each grid point
    lbl = np.argmin(dists, axis=0).reshape(X.shape)
    # Imprint labels on to grid
    positionLabel = lbl.copy()
    # Use a loop to map labels to unique identifiers
    for i in range(len(pointsIDs)):
        positionLabel[lbl == i] = pointsIDs[i]
    return positionLabel
      
# --- Save and load functions --- #

def pickle_Load(objectFilePath, printLoad = 1):
    """
    Loads and returns an object from specified filepath
    """
    if printLoad:
        print(f"Loading: {objectFilePath}")
    if os.path.exists(objectFilePath): 
        with open(objectFilePath, 'rb') as inp:
            obj = pickle.load(inp)
    else:
        print("Object file does not exist")
        return 0
    return obj

def pickle_Save(filepath, obj):
    """
    Saves object to some filepath using pickle.dump
    """
    # Check if file extension is provided, add one if not
    _, ext = os.path.splitext(filepath)
    if not ext:
        filepath += '.pkl'
        print('No file extension provided - adding .pkl extension')
    
    check_Create_Filepath(filepath)    
    with open(filepath, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    
    return 0

def check_Create_Filepath(filepath):
    """
    Checks if a filepath exists, if it does not - create it and print
    out that file path was created
    """
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory) and directory != '':
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        print(f"Created file structure for: {filepath}")
    return 0

# --- Space object utility functions --- #
def get_Upper_Cell_Size_Voxels():
    """ Returns dimension in voxels of cell size, should be based on available 
    computing power. It is linked to grain size """
    return 125

def get_Buffer(cellSize):
    """ Identifies how much of each cell to remove to ensure space is coherently
    generated """
    return get_Upper_Cell_Size_Voxels() - cellSize

def get_Cell_Space_Size(radius, targetSize):
    """
    Calculate the size that each cell is based on target grain radius and 
    a pre-set hard maxiumum processing limit for individual cells
    Args:
        radius (int): target grain radius in voxels
    Returns:
        cellSize (int): cell size in voxels
    """
    # Maximum cell size is function of seed density, check if below that threshold
    # if not, calculate smallest suitable cell size
    thresh = get_Upper_Cell_Size_Voxels()
    if max(targetSize) <= thresh:
        return max(targetSize), targetSize, [1,1,1]
    maxSize = get_Upper_Cell_Size_Voxels()
    # Cells should have an overlap to ensure cells are coherent, based on grain
    # radius. From grain size distriubtion very few grain should have more than
    # 2 times the average grain radius, so go bigger to be certain
    grainSizeBuffer = int(radius*2)
    cellSizeMax = maxSize - grainSizeBuffer
    numCells = [int(np.ceil(size / cellSizeMax)) for size in targetSize]
    # Correct to smallest possible cell size that won't involve creating more
    # cells - only relevant for small spaces
    cellSize = 0
    for i in range(3):
        cellSize = max(int(np.ceil(targetSize[i]/numCells[i])), cellSize)
    cellSize = min(cellSizeMax, cellSize)
    # Update space size for new cell size    
    spaceSize = [item * cellSize for item in numCells] 
    if cellSize < 0:
        print("Cell size is less than zero.")
    return cellSize, spaceSize, numCells

def get_Num_Cents(spaceDims, grainRad):
    """
    # Function to get the number of centers needed in the space for a target
    grain size, adjsuted for the extra space buffer that will be used and for
    the fraction of voxels used as boundaries
    Args:
        spaceSize (list): [x, y, z] integers for the size of space in voxels,
        with buffer applied
        grainRad (float): average target radius for each grain in voxels
    Returns: (int) number of grains to include in the space
    """
    # Calculate total space volume
    totalVol = 1
    for dim in spaceDims:
        # avoid using prod for multiplication to avoid overflow and negatives
        totalVol = totalVol*dim
    # Calculate approximate fraction of space covered by grain boundaries.
    ratioBoundary = ratio_Boundary(grainRad)
    if ratioBoundary > 0.40:
        print("The given grain radius is so small that the space won't generate properly")
    # Calculate volume available for grain interiors
    availableVol = (1-ratioBoundary)*totalVol
    # Calculate number of grains
    numGrains = math.ceil(availableVol/radius_To_Volume(grainRad))
    return numGrains

# --- Histogram / probabilities function --- #
def plot_Probdist(probs, ax, label = None, colour = 'b', labelAxs = 1):
    """ Plots and labels probability distribution on a log x-scale, 
    separating out P(0). """
    break_pos = 1e-4
    if label != None:
        ax.plot(probs[1][1:], probs[0][1:], color = colour, linewidth = 4,
                label = 'Distribution' + label)
        ax.scatter(break_pos, probs[0][0], s=400, marker='+', linewidths=3, color=colour,
                   label= 'Zero intensity' + label )
        ax.legend(fontsize = 30, loc = "upper left")
    else:
        ax.plot(probs[1][1:], probs[0][1:], color = colour, linewidth = 4)
    ax.scatter(break_pos, probs[0][0], s=400, marker='+', linewidths=3, color=colour)
    # Setting x-axis to log scale with specific breaks
    ax.set_xscale('log')
    if labelAxs == 1:
        ax.set_xlabel('Sodium Intensity (counts)', fontsize = 40)
        ax.set_ylabel('Probability', fontsize = 40)
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.tick_params(axis='both', which='minor', labelsize=26)
    return ax

def plot_Loghist(hist, ax, label = "", colour = 'b'):   
    """ Plots a histogram with log-scaled x-axis"""
    ax.hist(hist[1][1:-1], bins=hist[1][1:], weights=hist[0][1:], color = [colour], edgecolor='black', label = label)
    break_pos = 1e-4
    ax.plot([break_pos, break_pos], [0, max(hist[0])], color='k', linestyle='--', linewidth=1)
    ax.bar(break_pos/2, hist[0][0], width=break_pos, color=colour)
    ax.set_xlabel('Sodium Intensity (counts)')
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    return ax

def plot_Hist(x, ax, colour='b'):
    """ Plots a histogram given x values"""
    x = [val for val in x if val <= 5]
    hist, bins = np.histogram(x, bins= int(np.sqrt(len(x))))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    norm_hist = hist / np.max(hist)
    ax.bar(bin_centers, norm_hist, width=(bins[1] - bins[0]), 
            align='center', color='k', alpha=0.7, label = "Grain size histogram")
    return ax

def get_Loghist(x, bins = None):
    """ Function to create a log-scaled histogram of input x variable 
    Parameters:
        x [array-like]: variable x values
        bins [array-like]: number of bins
    Returns: [frequencies, bins], including zero intensity value as first item
    """
    # Auto-generate number of bins if not provided
    if bins == None:
        bins = int(np.sqrt(len(x)))
    # As data is always >= 0, to avoid taking log(0), deal with data close to
    # zero separately. Threshold is set based on ICP-MS sensitivity
    zeroThresh = 1e-4
    # Select points below threshold
    zeroValues = [point for point in x if point < zeroThresh]
    # Remove points below threshold from original array
    x = [point for point in x if point >= zeroThresh]
    # Calculate non-log bins
    _, bins = np.histogram(x, bins=bins)
    # Trensform bins to log space & re-calculate histogram
    # Apply offset, to avoid division by zero but still span data space
    bins = np.logspace(np.log10(bins[0]),
                       np.log10(bins[-1]), 
                       len(bins))
    freq, _ = np.histogram(x, bins=bins)
    return [np.insert(freq, 0, len(zeroValues)), np.insert(bins, 0, 0)]

def hist_To_Prob(hist):
    """ Takes histogram x values as intput and outputs normalised frequencies
    and original bin lower values to be used as a probability distirubtion
    """
    frequencies = hist[0]
    bins = hist[1] # these are bin edges 
    # If there is some probability of zero, must add this data on
    # Normalise the array
    arrSum = np.sum(frequencies)
    freqNormed = frequencies / arrSum
    return [freqNormed, bins[:-1]]


def plot_Expected_Grain_Dist(ax):
    """ Plot the expected distribution of grain sizes for a 3D Poission dist
    Equation is taken from https://doi.org/10.1016/j.physa.2007.07.063 """
    xs = np.linspace(0, 5, 500)
    ys = []
    for x in xs: 
        ys.append((3125/24)*(x*x*x*x)*math.exp((-5)*x))
    ax.plot(xs, ys, color = "#40E0D0", linewidth = 3, label = "Theoretical distribution")
    return 0

def graph_Profile(track, ax = None, label = "", orientation = "Vertical",
                  colour = "blue", supressTicks = 0, yStep = 1):
    """ Function to graph 1D data, used for empirical data """
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1) 
    if len(track) == 2:
        x = track[0]
        y = track[1]
    else:
        x = track
        y = [i for i in range(0, len(x)*yStep, yStep)]
    if orientation == "Vertical":
        ax.plot(x, y, color = colour, label = label, linewidth = 1)
    if orientation == "Horizontal":
        ax.plot(y, x, color = colour, label = label, linewidth = 1)
    if supressTicks:
        ax.set_yticks([])
    return ax

def plot_Cuboid(low, high, ax, edgeColor='k', toPlot="front"):
    """
    Plots a cuboid in a 3D space given lower and upper bounds. Used to help 
    visualise space limits during development

    Args:
        low (tuple): (x, y, z) coordinates of the lower bounds.
        high (tuple): (x, y, z) coordinates of the upper bounds.
        ax: The 3D axes object to plot on.
        edgeColor (str, optional): Color of the edges.
        toPlot (str, optional): Specifies which edges to plot. Options: "all", "front", or custom cases.
    """
    # Define cuboid limits
    xMin, yMin, zMin = low
    xMax, yMax, zMax = high

    # Define cuboid coordinates
    x = [xMin, xMin, xMax, xMax, xMin, xMin, xMax, xMax]
    y = [yMin, yMax, yMax, yMin, yMin, yMax, yMax, yMin]
    z = [zMin, zMin, zMin, zMin, zMax, zMax, zMax, zMax]

    # Define edges based on toPlot
    if toPlot == "all":
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                 (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    elif toPlot == "front":
        edges = [(2, 3), (3, 0), (4, 5), (5, 6),
                 (6, 7), (7, 4), (0, 4), (2, 6), (3, 7)]
    else:
        print("Invalid 'toPlot' option - using 'all'")
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                 (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    # Plot all edges of the cuboid
    for edge in edges:
        ax.plot([x[i] for i in edge], [y[i] for i in edge],
                [z[i] for i in edge], color=edgeColor, linewidth=4, zorder=10)
    return ax

def create_Cubes_And_Lims(positions,sizes=None,colors=None, **kwargs):
    """
    # Function to generate cubes at coordinates, based on code at
    https://stackoverflow.com/questions/49277753/plotting-cuboids
    """
    # Initialize variables to store the upper and lower limits
    lowerLims = positions[0].copy()
    upperLims = positions[0].copy()
    # Exceptions for if sizes, colours not passed to function
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    group = []
    # Create cubes with specified paramets
    for p,s,c in zip(positions,sizes,colors):
        group.append( cuboid_data(p, size=s))
        for i, coord in enumerate(p):
            lowerLims[i] = min(positions[0][i], coord)
            upperLims[i] = max(positions[0][i], coord)
    return (Poly3DCollection(np.concatenate(group), 
                             facecolors=np.repeat(colors,6, axis=0), **kwargs),
            lowerLims, upperLims)

# Function to generate cube axis coordinates
def cuboid_data(o, size=(1,1,1)):
    """
    # Returns info for plotting cube
    """
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

"""
def cuboidVerts(low, high, step):
    verticiesCoords = []
    # Calculate the number of steps needed in each dimension
    numStepsX = int(abs(high[0]-low[0])/step) + 1
    numStepsY = int(abs(high[1]-low[1])/step) + 1
    numStepsZ = int(abs(high[2]-low[2])/step) + 1
    # Generate the vertices
    for i in range(numStepsX):
        for j in range(numStepsY):
            for k in range(numStepsZ):
                x = low[0] + i*step
                y = low[1] + j*step
                z = low[2] + k*step
                verticiesCoords.append([x,y,z])
    return verticiesCoords
"""

def generate_Coordinates(low, high, mode="volume"):
    """
    Generates a list of coodinates in a cube for which voxels to plot.
    
    args:
        low, high (list): lists of low/high limits of cube in form (x,y,z)
        mode (str): Either "volume" to plot all voxels or "surface" to just
        plot voxels at surface of cube
    """
    xVals = np.arange(low[0], high[0] + 1)
    yVals = np.arange(low[1], high[1] + 1)
    zVals = np.arange(low[2], high[2] + 1)
    if mode == "volume":
        points = np.array(np.meshgrid(xVals, yVals, zVals)).T.reshape(-1, 3)
    elif mode == "surface":
        xMesh, yMesh, zMesh = np.meshgrid(xVals, yVals, zVals, indexing="ij")
        points = np.vstack([
            np.column_stack((xMesh[0].flatten(), yMesh[0].flatten(), zMesh[0].flatten())),
            np.column_stack((xMesh[-1].flatten(), yMesh[-1].flatten(), zMesh[-1].flatten())),
            np.column_stack((xMesh[:, 0].flatten(), yMesh[:, 0].flatten(), zMesh[:, 0].flatten())),
            np.column_stack((xMesh[:, -1].flatten(), yMesh[:, -1].flatten(), zMesh[:, -1].flatten())),
            np.column_stack((xMesh[:, :, 0].flatten(), yMesh[:, :, 0].flatten(), zMesh[:, :, 0].flatten())),
            np.column_stack((xMesh[:, :, -1].flatten(), yMesh[:, :, -1].flatten(), zMesh[:, :, -1].flatten())),
            np.column_stack((xMesh[0].flatten(), yMesh[0].flatten(), zMesh[:, 0].flatten())),
            np.column_stack((xMesh[0].flatten(), yMesh[-1].flatten(), zMesh[:, -1].flatten())),
            np.column_stack((xMesh[-1].flatten(), yMesh[0].flatten(), zMesh[:, 0].flatten())),
            np.column_stack((xMesh[-1].flatten(), yMesh[-1].flatten(), zMesh[:, -1].flatten()))
        ])
    else:
        print("For plotting mode must be either 'volume' to plot full 3D volume or 'surface' to plot only the surface voxels of the volume")
    return points.tolist()

# - Plot visuals - #
from math import floor, log10
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Formats input num to scientific notation - used in plots for cleaner visual
    """
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    return r"${0:.{2}f}×10^{{{1:d}}}$".format(coeff, exponent, precision)

def axis_Visuals(ax, yRight = 0, gridSpines = 1, xLabel = 1, yLabel = 1, 
                 legend = 1, legendTitle = None, log = 1, scale = 'cm'):
    #ax.set_ylim(ax.get_ylim()[::-1])
    ax.tick_params(axis='both', which='major', labelsize=30)
    if xLabel:
        label = scale.replace(" ", "")
        ax.set_xlabel(f'Relative model depth ({label})', fontsize=36)
    if yLabel:
        ax.set_ylabel('Sodium intensity (counts)', fontsize=36, labelpad = 30)
    if legend:
        ax.legend(title = legendTitle, title_fontsize=30, 
                  fontsize = 28, loc = "upper right")
    if log:
        ax.set_yscale('log')
    if scale == 'cm':
        scale_Micro_To_Centi(ax)
    if gridSpines:
        ax.grid(color='lightgray', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        #ax.spines['left'].set_visible(False)
    if yRight:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    return 0

def scale_Micro_To_Centi(ax, x = 1, y = 0):
    """ Converts micrometer values to centimeter values for plot """
    if x:
        ax.set_xticklabels([str(int(tick/10000)) for tick in ax.get_xticks()])
    if y:
        ax.set_yticklabels([str(int(tick/10000)) for tick in ax.get_yticks()])
    return 0


# --- Array manipulation --- #
def make_Mask(array, thresh):
    """ Returns a binary mask from an input array and specified threshold"""
    mask = array > thresh
    binary_mask = mask.astype(np.uint8)
    return binary_mask

def apply_Mask(img, mask):
    """Returns two one-dimensional lists of intensities corresponding
    to intensities in img located at positions of 0s and 1s in mask"""
    # Check if the dimensions of the arrays match
    if len(img) != len(mask) or len(img[0]) != len(mask[0]):
        print("Mask and input image should have the same dimensions")
        return 0
    boundaryIntensities = []
    interiorIntensities = []
    # Iterate over each element and apply mask
    for i in range(len(img)):
        for j in range(len(img[0])):
            if mask[i][j] == 0:
                interiorIntensities.append(img[i][j])
            elif mask[i][j] == 1:
                boundaryIntensities.append(img[i][j])
            else:
                print("Mask should be binary")
                return 0
    return boundaryIntensities, interiorIntensities

# --- Other functions --- #
def get_Seed():
    """Returns an integer, intended to be used to fix seed used for random 
    generation aspects of the code to allow reproducable results across 
    different runs"""
    return 3

def LA_Fill_Helper(args):
    """Helper function to allow parallelisation of cell creation - currently
    not implemented"""
    cellIndex, profiles, spaceObj = args
    return spaceObj.index_Cell_Fill(cellIndex, profiles)

def ratio_Boundary(avGrainRad):
    """Calculate the ratio of the boundary to interior voxels"""
    return ((avGrainRad + 1)**3 - (avGrainRad**3)) / (avGrainRad**3)

def volume_To_Radius(volume):
    """Calculate the radius of a sphere given its volume"""
    return (3 * volume / (4 * math.pi)) ** (1/3)

def radius_To_Volume(radius):
    """Calculate the volume of a sphere given its radius"""
    return (4/3) * math.pi * radius ** 3

def coord_Match(coord1, coord2):
    """Checks if 3d coords share a coordinate, returns coord index if yes, 
    -1 otherwise """
    for i, (c1, c2) in enumerate(zip(coord1, coord2)):
        if c1 == c2:
            return i  # Return the index of the matching coordinate
    return -1  # Return -1 if no matching coordinate is found

# get a list of colours
def get_Colours():
    """
    Function to get a list of predefined hex codes to use as colour inputs
    Returns:
        colourHexes: List of colour hex codes
    """
    colourList = [
            "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
            "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
            "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
            "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
            "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
            "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
            "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
            "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
            "#FF4A46", "#008941", "#006FA6", "#1CE6FF", "#FF34FF", "#A30059",
            "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
            "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
            "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
            "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
            "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
            "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
            "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]

    return colourList

# --- CSV Functions - for lines and maps --- #
# Function to convert csv names to plaintext names
def channelName(inputName):
    """
    Converts default channel names to formatted text
    """
    channelsDict = {
        '[0TIC]+': 'Total ion content',
        '[17O]+': 'Oxygen',
        '[23Na]+': 'Sodium',
        '[24Mg]+': 'Magnesium',
        '[56Fe]+': 'Iron'}

    return channelsDict[inputName]

def clipNeg(im):
    """
    Function to clip negative intensities, sets all values less than 0 to 0
    """
    return np.where(im < 0, 0, im)

def upperThreshold(im, threshold):
    """
    Function to set any values above a threshold to the threshold value
    """
    return np.where(im > threshold, threshold, im)

def lowerThreshold(im, threshold):
    """
    Function to set any values below a threshold to 0
    """
    return np.where(im < threshold, 0, im)
    
def gaussianFilterTrack(fwhm, originalTrack):
    """
    Apply a gaussian filter with a specified fwhm to an input 1d array
    """
    if fwhm == 1:
        return originalTrack[1:-1]
    else:
        # Set k (number of points in gauss) to be 2.5*fwhm, which is about 5 
        # sds from gauss mean. This is a reasonable width of gauss.
        k = int(fwhm*2.5)
        # Mirror signal at edge, it needs k extra samples at either edge in
        # order to mitigate edge effects 
        toApp = originalTrack[-k::]
        newWidths = np.append(originalTrack, toApp[::-1])
        toApp = originalTrack[:k]
        newWidths = np.insert(newWidths, 0, toApp[::-1])
        # Create gaussian filter to model experimental LA collection
        gaussTime = np.arange(-k,k)
        # Create Gaussian amplitudes
        gaussAmp = np.exp( -(4*np.log(2)*gaussTime**2) / fwhm**2 ) # fraction 
        # of points at each location to include in sum as gaussian is convoluted 
        # with signal
        
        # Convolve and plot smoothed functions
        # Initialize filtered signal vector
        gaussianFiltered = np.zeros(len(newWidths))
        # # implement the running mean filter
        for i in range(k+1,len(newWidths)-k-1):
            # each point is the weighted average of k surrounding points
            gaussianFiltered[i] = np.sum(newWidths[i-k:i+k]*gaussAmp) / sum(gaussAmp)
        # Return data, with cropped array to remove previous padding
        return  gaussianFiltered[k:-k]

# CSV Plotting functions #
def plot_Map(im, ax = None, cmap = 'inferno', extent = None, cBar = 0):
    """
    Plot a 2d map
    """
    # Plot original Image
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1) 
    img = ax.imshow(im, cmap=cmap, interpolation='nearest', extent = extent)
    if cBar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0)
        cbar = plt.colorbar(img, cax=cax)
        cbar.set_label('Sodium intenstiy', fontsize=26, labelpad = 10)  # Set the label and adjust the font size
        cbar.ax.tick_params(axis='y', labelsize=22)  # Adjust the labelsize as needed
        cbar.ax.ticklabel_format(useOffset=False, style='plain')

    return 0
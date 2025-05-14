This repository is for the framework discussed in the manuscript _What does the impurity variability at the microscale represent in ice cores? Insights from a conceptual approach_ , hosted at: https://doi.org/10.5194/tc-19-1373-2025

## Use summary

# Model generation routine

To generate a modelled ice volume run the script '_01_generateCells'. This script requires the following inputs, which are passed into the code by reading a saved JSON file, or through directly entering the parameters into the code:

name = 'EDC_514_1_20231115' # Name that modelled data will be saved under  
average_Grain_Radius = 500  # Target modelled grain radius - micrometers  
ice_Dimensions = [4000, 4000, 4000] # Target dimension x,y,z in micrometers  
chemical_File = 'chemical' # Chemical data file name  
chemical_Name = "Sodium" # Name of modelled impurity - only used for labels  
chemical_Threshold = 1000000 # Intensity threshold for 2d VISUALISATIONS  
chemical_Pixel_Size = 40 # LAICPMS spot size used to collect chemical data  
mask_File = 'mask' # Name of mask file

# Example cases and use cases used in publication

As an example case, if the codebase has been downloaded in its standard form, it's worth running the code using the data_From_File() function, calling the file "Inputs/Test/ice_data.JSON" to see the code work. Note that generating the real ice volumes can take a very long time, up to multiple days, even if run on a powerful machine.

The following files can be called to generate the modells discussed in the paper:  
"Inputs/EDC_514_1_20231115/ice_data.JSON" - EDC Holocene ice  
"Inputs/EDC_1994_3_20231114/ice_data.JSON" - EDC Last Glacial ice  
"Inputs/RECAP901_3_20230512/ice_data.JSON" - RECAP Holocene ice  
"Inputs/RECAP976_7_20230511/ice_data.JSON" - RECAP Last Glacial ice  
"Inputs/Grain_Sizes_Demo/ice_data.JSON" - Grain size exploration model discussed in Appendix A

# Model analysis routine
Once generated, the models can be analysed using the code '_02_Analyse' which is capable of returning experimental and modelled plots used in the manuscript









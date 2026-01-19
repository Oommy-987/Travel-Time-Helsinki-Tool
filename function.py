#========================================================================================
# File structure
#========================================================================================
# 1. Import libraries and modules
# 2. Define path
# 3. Global variables
# 3. Function 1: Interactive YKR_grid index map
# 4. Function 2-28: Functions used in Access viz tool
# 5. Function 29: Access viz tool

#========================================================================================
# Import libraries and modules
#========================================================================================
import geopandas                            
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt                  # for static map
import contextily                                # for static map and shortest possible path 
import folium                                    # for interactive map
from folium import Map, Element                  # for interactive map
import mapclassify                               # for interactive map
from IPython.display import display              # for interactive map
import osmnx as ox                               # for static map and shortest possible path 
from shapely.geometry import Point               # for static map and shortest possible path 

#========================================================================================
# Define path
#========================================================================================
import pathlib    
NOTEBOOK_PATH = pathlib.Path().resolve()
DATA_DIRECTORY = NOTEBOOK_PATH 

#========================================================================================
# Global variable
#========================================================================================
CODE_DICTIONARY ={
    "walk_t": "walk",
    "walk_d": "walk",
    "bike_f_t": "fast biking",
    "bike_s_t": "slow biking",
    "bike_d": "cycling route",
    "pt_r_tt": "public transport in rush hour + home waiting time",
    "pt_r_t": "public transport in rush hour",
    "pt_r_d": "public transportation route in rush hour",
    "pt_m_tt": "public transport in midday + home waiting time",
    "pt_m_t": "public transport in midday",
    "pt_m_d": "public transportation route in midday",
    "car_r_t": "car in rush hour", 
    "car_r_d": "car route in rush hour",
    "car_m_t": "car in midday",
    "car_m_d": "car route in midday",
    "car_sl_t": "car following speed limit"
}

YEAR_CODE_DICTIONARY = {
    "2018": [
        "walk_t","walk_d",
        "bike_f_t","bike_s_t","bike_d",
        "pt_r_tt","pt_r_t","pt_r_d","pt_m_tt","pt_m_t","pt_m_d",
        "car_r_t","car_r_d","car_m_t","car_m_d","car_sl_t"
    ],
    "2015": [
        "walk_t","walk_d",
        "pt_r_tt","pt_r_t","pt_r_d","pt_m_tt","pt_m_t","pt_m_d",
        "car_r_t","car_r_d","car_m_t","car_m_d"
    ],
    "2013": [
        "walk_t","walk_d",
        "pt_m_tt","pt_m_t","pt_m_d",
        "car_m_t","car_m_d"
    ]
}  
    
#========================================================================================
# Function 1: Interactive YKR_grid index map
#========================================================================================
def get_YKR():
    """
    Create an interactive YKR_grid index map
    """
    
    # Access the grid file
    ykr_grid=geopandas.read_file(
        DATA_DIRECTORY / 
        "data" / 
        "MetropAccess_YKR_grid")
    
    # Reproject to match CRS of OpenStreetMap
    ykr_grid = crs_conversion(file=ykr_grid,map_type="interactive")

    # Plot interactive map
    interactive_map = folium.Map(
        location = (60.2, 24.8),
        zoom_start = 10,
        control_scale = True,
        tiles=None
    )

    # Add the basemap
    folium.TileLayer(
        tiles="OpenStreetMap",
        opacity=0.5,
        attr=(
            "MetropAccess-YKR-grid: Digital Geography Lab; "
            "Basemap: OpenStreetMap contributors; "
            "CRS: EPSG:4326"
        )
    ).add_to(interactive_map)

    # Define the style of the YKR grid
    def style_function_ykr(feature):
        return {
            "fillColor": "transparent",
            "Color": "black",
            "weight": 2
        }

    # Add YKR_grid 
    folium.GeoJson(
        ykr_grid,
        style_function=style_function_ykr         
    ).add_to(interactive_map)
    
    # Create YKR_grid tooltips format
    tooltip = folium.features.GeoJsonTooltip(
        fields=("YKR_ID",),
        aliases=("YKR_ID:",)
    )

    # Create YKR_grid tooltips 
    tooltip_layer = folium.features.GeoJson(
        ykr_grid,
        style_function=style_function_tooltip,    
        tooltip=tooltip
    )

    # Add YKR_grid tooltips 
    tooltip_layer.add_to(interactive_map)   

    return interactive_map

#========================================================================================
# Function 2: Convert the CRS of the data file according to the map type
#========================================================================================
def crs_conversion(file,map_type):
    """
    Convect CRS of the data file according to the map type.
    
    Input:
    - file: geodataframe
    - map_type: str
        - Format: "static" or "interactive"
    
    Output: 
    - geodataframe with a suitable CRS
    """
    
    crs_map = {
        "static": "EPSG:3857",
        "interactive": "EPSG:4326"
    }
    
    return file.to_crs(crs_map[map_type])

#========================================================================================
# Function 3 Extract the centroid of the YKR ID destination grid 
#========================================================================================
def extract_destination_centroid(df, ykr_id):
    """
    Extract the centroid of the YKR ID destination grid. The centroid would be displayed
    in the map

    Input:
    - df: geodataframe
    - ykr_id: str

    Output:
    - destination_center: centroid (Shapely Point) of YKR ID destination grid
    """

    # Save the YKR_ID destination row in a new geodataframe
    destination_df = df.loc[df["YKR_ID"] == int(ykr_id)]
    
    # Generate a centroid Shapely Point
    destination_center = destination_df.geometry.iloc[0].centroid
    
    return destination_center

#========================================================================================
# Function 4: User-defined classification bins
#========================================================================================
def add_zero_to_mapclassify(df, data_column):
    """
    Create a quantile classification scheme for mapping travel‑time values in interactive
    map. Insert an additional class break at 0 if the dataset contains both positive and 
    negative values.
    
    Input:
    - df: geodataframe
    - data_column: columns used for map visualization

    Output:
    - classification bins
    """
    
    # Dont show NaN value in interactive map
    df = df.dropna(subset=[data_column])
    
    # Create map classifyer
    classifier = mapclassify.Quantiles(
        y=df[data_column],
        k=5
    )

    # Extract the data max and min
    data_min = df[data_column].min()
    data_max = df[data_column].max()

    # Add max and min value to the bin 
    bins = [data_min] + classifier.bins.tolist()
    bins[-1] = data_max

    # Drop the NaN value 
    bins = [b for b in bins if pd.notnull(b)]

    # Remove duplicate (by converting to a set) and sort the bins
    bins = sorted(set(bins))

    # If the classify bin does not have zero and the data ranges across zero
    # Append zero
    if 0 not in bins and (data_min) < 0 and (data_max) > 0:
        bins.append(0)

    # sort the bins again
    bins= sorted(bins)    

    return bins

#========================================================================================
# Function 5: Define a function for the style of the tooltip for all interactive map
#========================================================================================
def style_function_tooltip(feature):
    return {"color": "transparent", "fillColor": "transparent"}

#========================================================================================
# Function 6: Extract YKR ID from file path
#========================================================================================
def split_path_ykr(path):
    """
    Extract YKR ID.
    e.g travel time data file (2018_5785642_target_travel_time)
    e.g travel time diffference data file e.g 2018_6016691_bike_s_t_VS_2015_6016691_pt_m_t
    
    Input:
    - path: list or Path object

    Output:
    - ykr_id: str
    """
    
    ykr_id = path.stem[5:12]   # Slice the filename

    return ykr_id

#========================================================================================
# Function 7: Extract year from file path
#========================================================================================
def split_path_year(path):
    """
    Extract the year.
    e.g travel time data file (2018_5785642_target_travel_time)
    e.g travel time diffference data file e.g 2018_6016691_bike_s_t_VS_2015_6016691_pt_m_t
    
    Input:
    - path: list or Path object

    Output:
    - year: str
    """
   
    year = path.stem[0:4]    # Slice the filename
 
    return year

#========================================================================================
# Function 8: Travel time/distance difference contained in the same file
#========================================================================================
def compare_t2t_d2d(grid_travel_data,mode_cmp):
    """
    Compute travel time or distance differences between transport modes contained in the 
    same file.

    Input:
    - grid_travel_data: geodataframe
    - mode_cmp: list 
        - List of exactly two different travel time OR two different distance categories
        - Format: ["bike_s_t","pt_m_t"] 

    Output:
    - grid_travel_data: geodataframe
    
    Note:
    - the first item is always subtracted by the last one
    """

    # Extract the items
    mode_1, mode_2 = mode_cmp

    # Calculate the difference
    grid_travel_data[f"{mode_1}_VS_{mode_2}"] = (
        grid_travel_data[mode_1]-grid_travel_data[mode_2]
    )

    return grid_travel_data

#========================================================================================
# Function 9: Travel time difference between files from two different years
#========================================================================================
def compare_grids_t2t(grid_cmp,vector_type,vector_output_folder,grid_lookup):
    """
    Compute travel time difference between files from two different years.

    Input:
    - grid_cmp: list
        - List of two items: year_YKR ID_travel time category
        - Format: ["2018_6016691_bike_s_t","2015_6016691_pt_m_t"]
    - vector_type: str
        - Format: either "shp" for Shapefile or "gpkg" for Geopackage
    - vector_output_folder: str
        - Folder path where output files will be saved
    - grid_lookup: dictionary
        - Storing the found file path of the travel time data 
    
    Output:
    - grid_travel_data_cmp: geodataframe
    - filepath: Path object

    Note:
    - the first item is always subtracted by the last one
    """
    
    # Define two list to store the year_ykr and travel time type
    year_ykr_cmp = []
    cat_cmp = []

    # Extract the item
    grid_1, grid_2 = grid_cmp
    
    # For each items in grid_cmp
    # Extract the ykr_id and catergory
    for i in grid_cmp:
        year_ykr_id = i[:12]          # e.g "2018_6016691"
        catergory = i[13:]            # e.g "bike_s_t"
        year_ykr_cmp.append(year_ykr_id)  
        cat_cmp.append(catergory)

    # Extract the file paths in the dictionary
    path_1 = grid_lookup[year_ykr_cmp[0]]
    path_2 = grid_lookup[year_ykr_cmp[1]]

    # Load the two files 
    df_grid_1 = geopandas.read_file(path_1)
    df_grid_2 = geopandas.read_file(path_2)

    # Replace -1 with NaN in each travel time grid
    df_grid_1 = df_grid_1.replace(-1, np.nan)
    df_grid_2 = df_grid_2.replace(-1, np.nan)

    # Create a new geodataframe to store the calculation
    grid_travel_data_cmp = df_grid_1[["YKR_ID", "geometry"]].copy()

    # Calculate the travel time difference
    grid_travel_data_cmp[f"{grid_1}_VS_{grid_2}"] =(
        df_grid_1[cat_cmp[0]]-df_grid_2[cat_cmp[1]]
    ) 

    # Create a Path object
    vector_output_folder = pathlib.Path(vector_output_folder)

    # Save as Shapefile or GeoPackage
    if vector_type == "shp":
        name = f"{grid_1}_VS_{grid_2}"
        grid_path_cmp = vector_output_folder / f"{name}.shp"   
    elif vector_type == "gpkg":
        name = f"{grid_1}_VS_{grid_2}"
        grid_path_cmp = vector_output_folder / f"{name}.gpkg"

    # Export
    grid_travel_data_cmp.to_file(grid_path_cmp)
    
    return grid_travel_data_cmp,grid_path_cmp

#========================================================================================
# Function 10: Create a list of the travel time file paths of interests 
#========================================================================================
def find_travel_time_files(year_ykr):
    """
    Create a list of the travel time file paths of interests
    
    Input:
    - year_ykr: list
        - List of year_YKR ID, formatted as "2018_5785640"
        
    Output:
    - found_path: list with the selected YKR ID txt files filepath
    """

    # Split "year_ykr" variable into two lists containing the year and YKR_ID
    years = [i[:4] for i in year_ykr]
    ykr_ids = [i[5:] for i in year_ykr]

    # Extract the requested year
    # Convert the list into a set to remove duplicate years
    # A set does not allow extracting the value based on index
    # Convert back to a list
    year_requested = list(set(years))

    # Define a list to store ALL travel time data file path
    all_files = []

    # Loop each requested year
    # Each year corresponds to one parent folder to search
    # Prevent search in unrequested year folder
    # Collect all travel time data file path using .rglob
    for year in year_requested:
        folder = DATA_DIRECTORY / "data" / f"HelsinkiRegion_TravelTimeMatrix{year}"
        all_files.extend(folder.rglob("*"))

    # Define a list to store the target file paths
    found_path = []
        
    # Extract the index (starting from 1) and corresponding YKR ID using enumerate
    # Loop each index and YKR ID
    for index, ykr_id in enumerate(ykr_ids, start=1):

        # Total no. of search
        total_no = len(ykr_ids)

        # Index in the "years" list start from 0, so need to minus 1
        # Extract the target year
        # Concatenate the target filename
        # Note: leave a spacing before {ykr_id}
        target_year = years[index-1]
        target_filename = f"travel_times_to_ {ykr_id}.txt"
        
        # Update user on search progress
        print(
            f"Processing file {target_filename} in {target_year}...\
            Progress: {index}/{total_no}"
        )
        
        # Search through ALL travel time data file paths
        matches = None
        for files in all_files: 
            
            # i.parts splits the full file path into components 
            # Only the subfolder with year and filename are important
            # Check if they match the target
            if (
            files.name == target_filename
            and f"HelsinkiRegion_TravelTimeMatrix{target_year}" in files.parts
            ):
                matches = files
                
        # If any matching paths were found, store them
        # If none, raise warning
        if matches:
            found_path.append(matches)
        else:
            print(f"No file found for YKR ID {ykr_id}")

    return found_path
                
#========================================================================================
# Function 11: Write the found_path to a txt file
#========================================================================================
def write_to_txt(found_path):
    """
    Write the found_path to a txt file.

    Input:
    - found_path: list for the selected YKR ID files filepath
    
    Output:
    - txt file listing file paths
    """
  
    # Create a Path object
    output_txt_name = pathlib.Path("found_path.txt")
        
    # Write the paths into the text file
    # Open the txt file; overwrite the file if it already exits
    # the "with" block close the file afterwards
    with open(output_txt_name, "w", encoding="utf-8") as f:
                
        # Loop each file path
        for fp in found_path:

            # Convert the path in string and write one path per line
            f.write(str(fp) + "\n") 
            
    # Notify user that filepath was written in the txt file
    print(f"Filepaths saved in: {output_txt_name}")
            
#========================================================================================
# Function 12: Join the travel time data into YKR_grid.shp and export the spatial file
#========================================================================================
def join_travel_time_table (year_ykr,vector_type,vector_output_folder):
    """
    Join the travel time data into YKR_grid.shp and export the vector file.
    
    Input:
    - year_ykr: list
        - List of year_YKR ID, formatted as "2018_5785640"
    - vector_type: str
        - Format: either "shp" for Shapefile or "gpkg" for Geopackage
    - vector_output_folder: str
        - Folder path where output files will be saved

    Output:
    - grid_path: Path 
        - file path for travel time data spatial layer
    """

    # Collect the travel time data file path for selected YKR ID
    found_path = find_travel_time_files(year_ykr)
     
    # Access the grid file
    ykr_grid = geopandas.read_file(
        DATA_DIRECTORY / 
        "data" / 
        "MetropAccess_YKR_grid" / 
        "MetropAccess_YKR_grid_EurefFIN.shp"
    )
    
    # Define a empty list to store the output spatial file path
    grid_path = []
    
    # Loop each travel time data file
    for path, name in zip (found_path,year_ykr):
        
        # Load the travel time data file
        data = pd.read_csv(path, sep=";")
        
        # Join with the YKR_grid.shp
        joined = ykr_grid.merge(
            data, 
            how="left", 
            left_on="YKR_ID", 
            right_on="from_id"
        )
        
        # Save as Shapefile or GeoPackage
        if vector_type == "shp":
            output_path = vector_output_folder / f"{name}_target_travel_time.shp"
        elif vector_type == "gpkg":
            output_path = vector_output_folder / f"{name}_target_travel_time.gpkg"
        
        # Append the filepath to a list
        grid_path.append(output_path)
    
        # Export 
        if vector_type and vector_output_folder:
            joined.to_file(output_path)

    return grid_path

#========================================================================================
# Function 13: HSL zone processing for map visualization
#========================================================================================
def hsl_zone_processing(map_type):
    """
    Load HSL transif zone and convert the CRS according to the map type. Zone D has been
    deleted in the original file for visualization purpose.

    Input:
    - map_type: str
        - Format: "static" or "interactive"

    Output:
    - hsl_zone geodataframe with a suitable CRS
    """

    # Load the gpkg
    hsl_zone = geopandas.read_file(DATA_DIRECTORY / "data" / "HSL Transif Zone.gpkg")
    
    # CRS for static map: EPSG:3857
    # CRS for interactive map: EPSG:4326
    hsl_zone = crs_conversion(file=hsl_zone,map_type=map_type)

    return hsl_zone
    
#========================================================================================
# Function 14: Travel time data file processing for map visualization 
#========================================================================================
def map_data_processing(path,map_type):
    """
    Data processing for map visulaization
    - Load the travel time spatial file 
    - Replace -1 with NaN in file, if any
    - Convert the CRS of the file to one suitable for static or interactive maps
    - Create a centroid point of the YKR ID destination marker (shown in map)

    Input:
    - path: list or Path object
    - map_type: str
        - Format: "static" or "interactive"

    Output: 
    - df: geodataframe
    - hsl_zone: geodataframe
    - destination_center: centroid (Shapely Point) of YKR ID destination grid
    """

    # Load the HSL zone
    hsl_zone = hsl_zone_processing(map_type)

    # Load the travel time data spatial file
    df = geopandas.read_file(path)
    
    # Replace -1 with NaN if any
    if (df == -1).any().any():
        df = df.replace(-1, np.nan)

    # CRS for static map: EPSG:3857
    # CRS for interactive map: EPSG:4326
    df = crs_conversion(file=df,map_type=map_type)

    # Extract the YKR ID destination from the filename
    ykr_id = split_path_ykr(path)

    # Generate a centroid (Shapely Point) of the YKR ID destination marker
    destination_center = extract_destination_centroid(df=df, ykr_id=ykr_id)

    return df,destination_center,hsl_zone

#========================================================================================
# Function 15: Create travel time data classification in static map
#========================================================================================
def static_class(df,column):
    """
    Create travel time data classification in static map
    
    Input:
    - df: geodataframe to be plotted
    - column: column used to plot in the dataframe

    Output:
    - static map layout
    """
    
    static_map = df.plot(
        figsize=(12, 8),
        column=column,
        scheme="quantiles",          
        k=5,                         
        cmap="Spectral",
        linewidth=0,
        alpha=0.6,
        legend=True,
    )

    return static_map

#========================================================================================
# Function 16: Create travel time/distance data difference classification in static map 
#========================================================================================
def static_class_cmp(df,column):
    """
    Plot static map layout for displaying travel time/distance difference. Compared to 
    Function 15, this function can force a class break at zero if the data range cross 
    zero. This manual break is useful in map showing positive and negative difference.
    
    Input:
    - df: geodataframe to be plotted
    - column: column used to plot in the dataframe

    Output:
    - static map layout
    """

    # Create a Quantiles classifer
    classifer = mapclassify.Quantiles(y=df[column],k=5)
    
    # Extract the bin 
    q = classifer.bins.tolist()

    # Drop the NaN value
    q = [b for b in q if pd.notnull(b)]

    # Remove duplicate and sort the bins
    q = sorted(set(q))

    # Extract max and min data value
    data_min = df[column].min()
    data_max = df[column].max()
    
    # If the classify bin does not have zero and the data ranges across zero
    # Append zero
    if 0 not in q and (data_min) < 0 and (data_max) > 0:
        q.append(0)

    # Sort
    q= sorted(q)

    # Plot
    static_map = df.plot(
        figsize=(12, 8),
        column=column,
        scheme="UserDefined", 
        classification_kwds={"bins": q},                    
        cmap="Spectral",
        linewidth=0,
        alpha=0.6,
        legend=True,
    )

    return static_map
    
#========================================================================================
# Function 17: Plot common elements in static map
#========================================================================================
def static_common_element(
    df_path,static_map,
    destination_center,hsl_zone,
    figure_title,legend_title
):
    """
    Plot all common elements in static map
    - YKR ID destination marker
    - HSL zone
    - Basemap
    - Figure title
    - Figure extent
    
    Input:
    - df_path: list or Path
    - static_map
    - destination_center: centroid (Shapely Point) of YKR ID destination grid
    - hsl_zone: geodataframe
    - figure_title: str

    Output:
    - static map
    """

    # Common element - YKR ID destination marker
    # Extract ykr_id
    ykr_id = str(split_path_ykr(df_path))
    
    # Mark the destination with a star
    static_map.scatter(
        destination_center.x, 
        destination_center.y,
        marker="*",
        s=120,
        facecolor="black",
        linewidth=0.8,
        zorder=60
    )

    # Annotate with YKR ID
    static_map.text(
        destination_center.x+1000,
        destination_center.y,
        ykr_id,
        fontsize=8,
        color="black",
        ha="left",
        va="top",
        zorder=65
    )
    
    # Common element - HSL zone
    hsl_zone.plot(
        ax=static_map,
        facecolor="none",
        edgecolor="royalblue",
        linewidth=1,
        zorder=25,
    )
    
    # Extract the letter of HSL zone
    # Note: Zone D has been deleted in the source file
    zone = hsl_zone["Zone"].astype(str).tolist()
    
    # Loop each HSL zone with the index
    for index, letter in enumerate(zone):
        
        # Generate representaive point for each zone
        # Note: centroid of the HSL zone lies outside of the polygon
        hsl_pt = hsl_zone.geometry.iloc[index].representative_point()
           
        # Plot the zone symbol
        static_map.text(
            hsl_pt.x,
            hsl_pt.y,
            letter,
            fontsize=20,
            fontweight="bold",
            color = "royalblue",
            zorder=50,
        )

    # Common element - Map extent
    # Retreive the map extent
    xlim = static_map.get_xlim()
    ylim = static_map.get_ylim()
        
    # Extend the limits by 5% more area
    x_margin = (xlim[1] - xlim[0]) * 0.05
    y_margin = (ylim[1] - ylim[0]) * 0.05

    # Apply the defined extent
    static_map.set_xlim(xlim[0] - x_margin, xlim[1] + x_margin)
    static_map.set_ylim(ylim[0] - y_margin, ylim[1] + y_margin)

    # Common element - Basemap
    contextily.add_basemap(
        static_map,
        source=contextily.providers.OpenStreetMap.Mapnik,
        alpha=0.5,
        attribution=(
            "Travel time data: Digital Geography Lab;\n"
            "Helsinki Region Transport transif zones: Helsinki Region Infoshare;\n"
            "Basemap: OpenStreetMap contributors; CRS: EPSG:3857"
        )
    )

    # Common element - Figure title
    static_map.set_title(figure_title)

    # Commmon element - legend title
    leg = static_map.get_legend()
    leg.set_loc("lower right")
    leg.set_title(legend_title)
    
    return static_map

#========================================================================================
# Function 18: Plot static map with the travel time data 
#========================================================================================
def plot_static_map_travel_time(grid_path,map_type,transport_mode):
    """
    Plot the travel time data in a static map

    Input:
    - grid_travel_data: geodataframe
    - map_type: str
        - Format: "static" or "interactive"
    - transport_mode: str
        - Transport mode of interest
    
    Output:
    - static map with the travel time data, annotated HSL zone and destination marker
    """

    # Retrieve the key componenet of the figure title
    label = CODE_DICTIONARY.get(transport_mode)
    ykr_id = split_path_ykr(grid_path)
    year = split_path_year(grid_path)

    # Concentrate the figure title
    figure_title = f"Travel Time ({year}) Map ({label}) To YKR ID {ykr_id}"

    # Data processing for map visualization
    grid_travel_data,destination_center,hsl_zone = (
    map_data_processing(
        path=grid_path,
        map_type=map_type
    ))

    # Apply the static map classiifcation scheme
    static_map_layout = static_class(
        df=grid_travel_data,
        column=transport_mode
    )

    # Finalize the static map
    static_map_travel_time = static_common_element(
        df_path=grid_path,
        static_map=static_map_layout,
        destination_center=destination_center,
        hsl_zone=hsl_zone,
        figure_title=figure_title,
        legend_title="Travel time (min)"
    )
    
    return static_map_travel_time

#========================================================================================
# Function 19: Plot static map showing the travel time/distance difference
#========================================================================================
def plot_static_map_mode_cmp(grid_path,map_type,mode_cmp):
    """
    Plot the travel time/distance difference in a static map

    Input:
    - grid_path: Path 
        - File path for travel time data spatial layer
    - map_type: str
        - Format: "static" or "interactive"
    - mode_cmp: list 
        - List of exactly two different travel time OR two different distance categories
        - Format: ["bike_s_t","pt_m_t"] 
    
    Output:
    - static map with the travel time/distance difference, annotated HSL zone 
    and destination marker
    """
    
    # Extract the two items in mode_cmp
    mode_1, mode_2 = mode_cmp

    # Data processing for map visualization
    grid_travel_data,destination_center,hsl_zone = (
    map_data_processing(
        path=grid_path,
        map_type=map_type
    ))
    
    # Calculate the time/distance difference
    grid_travel_data = compare_t2t_d2d(
        grid_travel_data=grid_travel_data,
        mode_cmp=mode_cmp
    )

    # Retrieve the key componenet of the figure title
    label_1 = CODE_DICTIONARY.get(mode_1)
    label_2 = CODE_DICTIONARY.get(mode_2)
    ykr_id = split_path_ykr(grid_path)
    year = split_path_year(grid_path)
    
    # Concentrate the figure and legend title for travel time difference
    if mode_1[-1:]== "t":
        figure_title = (
            f"Travel Time Difference ({year}) Map\n"
            f"({label_1} V.S {label_2}) To YKR ID {ykr_id}"
        )
        
        legend_title = "Travel time difference (min)"
        
    # Concentrate the figure and legend title for travel distance difference
    if mode_1[-1:]== "d":
        figure_title = (
            f"Travel Distance Difference ({year}) Map\n"
            f"({label_1} V.S {label_2}) To YKR ID {ykr_id}"
        )
        
        legend_title = "Travel distance difference (m)"

    # Apply the static map classiifcation scheme
    static_map_layout = static_class_cmp(
        df=grid_travel_data,
        column=f"{mode_1}_VS_{mode_2}"
    )

    # Finalize the static map
    static_map_mode_cmp = static_common_element(
        df_path=grid_path,
        static_map=static_map_layout,
        destination_center=destination_center,
        hsl_zone=hsl_zone,
        figure_title=figure_title,
        legend_title=legend_title
    )
        
    return static_map_mode_cmp

#========================================================================================
# Function 20: Plot static map with travel time difference between year
#========================================================================================
def plot_static_map_grid_cmp(
    grid_path_cmp,grid_cmp,
    map_type,vector_type,vector_output_folder,
    grid_lookup
):
    """
    Plot the travel time difference between years in a static map

    Input:
    - grid_path_cmp: Path 
        - File path of the travel time data difference spatial layer
    - map_type: str
        - Format: "static" or "interactive"
    - grid_cmp: list
        - List of two items: year_YKR ID_travel time category
        - Format: ["2018_6016691_bike_s_t","2015_6016691_pt_m_t"]
    - vector_type: str
        - Format: either "shp" for Shapefile or "gpkg" for Geopackage
    - vector_output_folder: str
        - Folder path where output files will be saved
    - grid_lookup: dictionary
        - Storing the found file path of the travel time data 
    
    Output:
    - static map with the travel time difference between years, annotated HSL zone 
    and destination marker
    """
    
    # Extract the two items in grid_cmp
    grid_1, grid_2 = grid_cmp

    # Calculate the travel time difference between two files
    grid_travel_data_cmp,grid_path_cmp = compare_grids_t2t(
        grid_cmp=grid_cmp,
        vector_output_folder=vector_output_folder,
        vector_type=vector_type,
        grid_lookup=grid_lookup
    )
    
    # Data processing before map visualization
    grid_travel_data_cmp,destination_center,hsl_zone = (
        map_data_processing(
            path=grid_path_cmp,
            map_type=map_type
        ))

    # Retrieve the key componenet of the figure title
    cmp_label_1 = CODE_DICTIONARY.get(grid_1[13:])  # e.g "bike_s_t"
    cmp_year_1 = grid_1[:4]                         # e.g "2018"
    cmp_ykr_id_1 = grid_1[5:12]                     # e.g :6016691"
    
    cmp_label_2 = CODE_DICTIONARY.get(grid_2[13:])  # e.g "bike_s_t"
    cmp_year_2 = grid_2[:4]                         # e.g "2018"

    # Concentrate the figure title
    figure_title = (
        f"Travel Time Difference ({cmp_ykr_id_1}) Map\n"
        f"({cmp_label_1} in {cmp_year_1} V.S {cmp_label_2} in {cmp_year_2})"
    )
    
    # Apply the static map classiifcation scheme
    static_map_layout = static_class_cmp(
        df=grid_travel_data_cmp,
        column=f"{grid_1}_VS_{grid_2}"
    )

    # Finalize the static map
    static_map_grid_cmp = static_common_element(
        df_path=grid_path_cmp,
        static_map=static_map_layout,
        destination_center=destination_center,
        hsl_zone=hsl_zone,
        figure_title=figure_title,
        legend_title="Travel time (min)"
    )

    return static_map_grid_cmp

#========================================================================================
# Function 21: Plot interactive map layout 
#========================================================================================  
def interactive_common_element(df_path,df,column,destination_center):
    """
    Plot the common element of interactive map.
    - YKR ID destination marker
    - Basemap
    
    Input:
    - df_path: list or Path
        Filepath of the geodataframe
    - df: geodataframe 
    - column: column used to plot in the dataframe
    - destination_center: centroid (Shapely Point) of YKR ID destination grid

    Output:
    - interactive map
    """

    # Extract the YKR ID
    ykr_id = split_path_ykr(df_path)

    # Common element - layout
    interactive_map = folium.Map(
        location = (60.2, 24.8),
        zoom_start = 10,
        control_scale = True,
        tiles=None
    )

    # Common element - YKR ID destination
    destination = folium.Marker(
        location = (destination_center.y,destination_center.x),
        tooltip = ykr_id,
        icon = folium.Icon(color="black", icon="star")
    )
    
    destination.add_to(interactive_map)

    # Common element - Basemap
    folium.TileLayer(
        tiles="OpenStreetMap",
        opacity=0.5,
        attr=(
            "Travel time data: Digital Geography Lab; "
            "Helsinki Region Transport tariff zones: Helsinki Region Infoshare;\n"
            "Basemap: OpenStreetMap contributors; "
            "CRS: EPSG:4326"
        )
    ).add_to(interactive_map)

    return interactive_map

#========================================================================================
# Function 22: Add the HSL zone on the topmost layer of the interactive map
#========================================================================================
def hsl_zone_top(interactive_map,hsl_zone):
    """
    Add the HSL zone on the topmost layer of the interactive map

    Input:
    - interactive_map: str
    - hsl_zone: geodataframe
    """
    
    # Common element - HSL zone
    folium.GeoJson(
        hsl_zone,
        name="HSL Zones",
        style_function=lambda feature: {
        "fillColor": "none",
        "color": "royalblue",
        "weight": 2,
        "fillOpacity": 0.1
        }
    ).add_to(interactive_map)

    # Extract the letter of HSL zone
    # Note: Zone D has been deleted in the source file
    zone = hsl_zone["Zone"].astype(str).tolist()
    
    # Loop over each HSL zone with the index 
    for index, letter in enumerate(zone):
        
        # Generate representaive point for each zone
        # Note: centroid of the HSL zone lies outside of the polygon
        hsl_pt = hsl_zone.geometry.iloc[index].representative_point()

        # Add the text marker
        folium.map.Marker(
            location=(hsl_pt.y, hsl_pt.x),
            icon=folium.DivIcon(
            html=f"""
                <div style="
                    font-size: 40px;
                    font-weight: bold;
                    color: royalblue;
                    text-align: center;
                ">
                    {letter}
                </div>
                """
            )
        ).add_to(interactive_map)

    return interactive_map
    
#========================================================================================
# Function 23: Plot interactive map with the travel time data 
#========================================================================================
def plot_interactive_map_travel_time(grid_path,map_type,transport_mode):
    """
    Plot the travel time data in a interactive map

    Input:
    - grid_path: Path 
        - File path for travel time data spatial layer
    - map_type: str
        - Format: "static" or "interactive"
    - transport_mode: str
        - Transport mode of interest
    
    Output:
    - interactive map with the travel time data, annotated HSL zone and destination marker
    """

    # Data processing for map visualization
    grid_travel_data,destination_center,hsl_zone = (
    map_data_processing(
        path=grid_path,
        map_type=map_type
    ))

    # Plot interactive map
    interactive_map_travel_time = interactive_common_element(
        df_path=grid_path,
        df=grid_travel_data,
        column=transport_mode,
        destination_center=destination_center
    )

    # Dont show NaN value in interactive map
    grid_travel_data = grid_travel_data.dropna(subset=[transport_mode])
    
    # Create the str index column
    grid_travel_data.loc[:, "id"] = grid_travel_data.index.astype(str)
        
    # Create map classifyer
    classifier = mapclassify.Quantiles(
        grid_travel_data[transport_mode]
    )
        
    # Add the min and max value of the data to the bin
    # Convert the bins to a list
    bins = [grid_travel_data[transport_mode].min()] + classifier.bins.tolist()
    bins[-1] = grid_travel_data[transport_mode].max()

    # Create Choropleth and add to the map
    grid_layer = folium.Choropleth(
        geo_data = grid_travel_data,
        data = grid_travel_data, 
        columns = ("id",transport_mode),
        key_on = "feature.id",
        bins=bins,
        fill_color="Spectral", 
        fill_opacity=0.7,
        legend_name="Travel time (min)"
    )
    
    grid_layer.add_to(interactive_map_travel_time)

    # Create travel time tooltips format
    tooltip = folium.features.GeoJsonTooltip(
        fields=(transport_mode,),
        aliases=("Travel time (min)",)
    )

    # Add the travel time data 
    tooltip_layer = folium.features.GeoJson(
        grid_travel_data,
        style_function=style_function_tooltip,
        tooltip=tooltip
    )

    tooltip_layer.add_to(interactive_map_travel_time)

    # Retrieve the key component of figure title
    label = CODE_DICTIONARY.get(transport_mode)
    ykr_id = split_path_ykr(grid_path)
    year = split_path_year(grid_path)

    # Add figure title
    title_html = f"""
                <h3 align="center" style="font-size:20px">
                    Travel Time ({year}) Map ({label}) to YKR ID {ykr_id}
                </h3>
            """
    interactive_map_travel_time.get_root().html.add_child(Element(title_html))

    # Add the HSL on top
    interactive_map_travel_time = hsl_zone_top(
        interactive_map=interactive_map_travel_time,
        hsl_zone=hsl_zone
    )
    
    return interactive_map_travel_time
    
#========================================================================================
# Function 24: Plot interactive map showing the travel time/distance difference
#========================================================================================
def plot_interactive_map_mode_cmp(grid_path,map_type,mode_cmp):
    """
    Plot the travel time/distance difference in a interactive map

    Input:
    - grid_path: Path
        - file path for travel time data spatial layer
    - map_type: str
        - Format: "static" or "interactive"
    - mode_cmp: list 
        - List of exactly two different travel time OR two different distance categories
        - Format: ["bike_s_t","pt_m_t"] 
    
    Output:
    - interactive map with the travel time/distance difference, annotated HSL zone 
    and destination marker
    """
    
    # Extract the two items in mode_cmp
    mode_1, mode_2 = mode_cmp

    # Retrieve the key componenet of the figure title
    label_1 = CODE_DICTIONARY.get(mode_1)
    label_2 = CODE_DICTIONARY.get(mode_2)
    ykr_id = split_path_ykr(grid_path)
    year = split_path_year(grid_path)
    
    # Data processing for map visualization
    grid_travel_data,destination_center,hsl_zone = (
    map_data_processing(
        path=grid_path,
        map_type=map_type
    ))
    
    # Calculate the time/distance difference
    grid_travel_data = compare_t2t_d2d(
        grid_travel_data=grid_travel_data,
        mode_cmp=mode_cmp
    )

    # Dont show NaN value in interactive map
    grid_travel_data = grid_travel_data.dropna(subset=[f"{mode_1}_VS_{mode_2}"])

    # Create the str index column
    grid_travel_data.loc[:, "id"] = grid_travel_data.index.astype(str)
    
    # Create the classifier    
    q = add_zero_to_mapclassify(
        df=grid_travel_data, 
        data_column=f"{mode_1}_VS_{mode_2}"
    )

    # Plot interactive map
    interactive_map_mode_cmp = interactive_common_element(
        df_path=grid_path,
        df=grid_travel_data,
        column=f"{mode_1}_VS_{mode_2}",
        destination_center=destination_center
    )
    
    # For time-time difference map
    if mode_1[-1:]== "t":

        # Create Choropleth and add to the map
        grid_layer = folium.Choropleth(
            geo_data = grid_travel_data,
            data = grid_travel_data, 
            columns = ("id",f"{mode_1}_VS_{mode_2}"),
            key_on = "feature.id",
            bins=q,
            fill_color="Spectral", 
            fill_opacity=0.7,
            legend_name="Travel time difference (min)"
        )
    
        grid_layer.add_to(interactive_map_mode_cmp)

        # Create travel time difference tooltips format
        tooltip = folium.features.GeoJsonTooltip(
            fields=(f"{mode_1}_VS_{mode_2}",),
            aliases=("Travel time difference (min)",)
        )

        # Add the travel time difference 
        tooltip_layer = folium.features.GeoJson(
            grid_travel_data,
            style_function=style_function_tooltip,
            tooltip=tooltip
        )

        tooltip_layer.add_to(interactive_map_mode_cmp)

        # Add figure title
        title_html = f"""
                    <h3 align="center" style="font-size:20px">
                        Travel Time Difference ({year}) Map ({label_1} V.S {label_2} To YKR ID {ykr_id})
                    </h3>
                """
        interactive_map_mode_cmp.get_root().html.add_child(Element(title_html))

    # For distance-distance difference map
    if mode_1[-1:]== "d":

        # Create Choropleth and add to the map
        grid_layer = folium.Choropleth(
            geo_data = grid_travel_data,
            data = grid_travel_data, 
            columns = ("id",f"{mode_1}_VS_{mode_2}"),
            key_on = "feature.id",
            bins=q,
            fill_color="Spectral", 
            fill_opacity=0.7,
            legend_name="Travel distance difference (m)"
        )
    
        grid_layer.add_to(interactive_map_mode_cmp)

        # Create travel distance difference tooltips format
        tooltip = folium.features.GeoJsonTooltip(
            fields=(f"{mode_1}_VS_{mode_2}",),
            aliases=("Travel distance difference (m)",)
        )

        # Add the travel distance difference 
        tooltip_layer = folium.features.GeoJson(
            grid_travel_data,
            style_function=style_function_tooltip,
            tooltip=tooltip
        )

        tooltip_layer.add_to(interactive_map_mode_cmp)

        # Add figure title
        title_html = f"""
                    <h3 align="center" style="font-size:20px">
                        Travel Distance Difference ({year}) Map ({label_1} V.S {label_2} To YKR ID {ykr_id})
                    </h3>
                """
        interactive_map_mode_cmp.get_root().html.add_child(Element(title_html))

    # Add the HSL on top
    interactive_map_mode_cmp = hsl_zone_top(
        interactive_map=interactive_map_mode_cmp,
        hsl_zone=hsl_zone
    )
    
    return interactive_map_mode_cmp

#========================================================================================
# Function 25: Plot interactive map with travel time difference between year
#========================================================================================
def plot_interactive_map_grid_cmp(
    grid_path_cmp,map_type,grid_cmp,
    vector_type,vector_output_folder,
    grid_lookup
):
    """
    Plot the travel time difference between yea in a interactive map

    Input:
    - grid_path_cmp: Path 
        File path of the travel time data difference spatial layer
    - map_type: str
        - Format: "static" or "interactive"
    - grid_cmp: list
        - List of two items: year_YKR ID_travel time category
        - Format: ["2018_6016691_bike_s_t","2015_6016691_pt_m_t"]
    - vector_type: str
        - Format: either "shp" for Shapefile or "gpkg" for Geopackage
    - vector_output_folder: str
        - Folder path where output files will be saved
    - grid_lookup: dictionary
        - Storing the found file path of the travel time data 
    
    Output:
    - interactive map with the travel time difference between years, annotated HSL zone 
    and destination marker
    
    """
    # Extract the two items in grid_cmp
    grid_1, grid_2 = grid_cmp

    # Extract the YKR ID and year
    ykr_id = split_path_ykr(grid_path_cmp)
    year = split_path_year(grid_path_cmp)

    # Retrieve the key componenet of the figure title
    cmp_label_1 = CODE_DICTIONARY.get(grid_1[13:]) # e.g "bike_s_t"
    cmp_year_1 = grid_1[:4]                        # e.g "2018"
    cmp_ykr_id_1 = grid_1[5:12]                    # e.g :6016691"
    
    cmp_label_2 = CODE_DICTIONARY.get(grid_2[13:]) # e.g "bike_s_t"
    cmp_year_2 = grid_2[:4]                        # e.g "2018"
    
    # Calculate the travel time difference between two files
    grid_travel_data_cmp,grid_path_cmp = compare_grids_t2t(
        grid_cmp=grid_cmp,
        vector_output_folder=vector_output_folder,
        vector_type=vector_type,
        grid_lookup=grid_lookup
    )
    
    # Data processing for map visualization
    grid_travel_data_cmp,destination_center,hsl_zone = (
        map_data_processing(
            path=grid_path_cmp,
            map_type=map_type
        ))
    
    # Dont show NaN value in interactive map
    grid_travel_data_cmp = grid_travel_data_cmp.dropna(subset=[f"{grid_1}_VS_{grid_2}"])
    
    # Plot interactive map
    interactive_map_grid_cmp = interactive_common_element(
        df_path=grid_path_cmp,
        df=grid_travel_data_cmp,
        column=f"{grid_1}_VS_{grid_2}",
        destination_center=destination_center
    )

    # Create the str index column
    grid_travel_data_cmp.loc[:, "id"] = grid_travel_data_cmp.index.astype(str)

    # Create the classifier
    q = add_zero_to_mapclassify(
        df=grid_travel_data_cmp, 
        data_column=f"{grid_1}_VS_{grid_2}"
    )

    # Create Choropleth and add to the map
    grid_layer = folium.Choropleth(
        geo_data = grid_travel_data_cmp,
        data = grid_travel_data_cmp, 
        columns = ("id",f"{grid_1}_VS_{grid_2}"),
        key_on = "feature.id",
        bins=q,
        fill_color="Spectral", 
        fill_opacity=0.7,
        legend_name="Travel time (min)"
    )
    
    grid_layer.add_to(interactive_map_grid_cmp)

    # Create travel time tooltips format
    tooltip = folium.features.GeoJsonTooltip(
        fields=(f"{grid_1}_VS_{grid_2}",),
        aliases=("Travel time difference (min)",)
    )

    tooltip_layer = folium.features.GeoJson(
        grid_travel_data_cmp,
        style_function=style_function_tooltip,
        tooltip=tooltip
    )

    tooltip_layer.add_to(interactive_map_grid_cmp)

    # Add figure title
    title_html = f"""
                    <h3 align="center" style="font-size:20px">
                        Travel Distance Difference ({year}) Map ({cmp_label_1} V.S {cmp_label_2} To YKR ID {ykr_id})
                    </h3>
                """
    interactive_map_grid_cmp.get_root().html.add_child(Element(title_html))

    # Add the HSL on top
    interactive_map_grid_cmp = hsl_zone_top(
        interactive_map=interactive_map_grid_cmp,
        hsl_zone=hsl_zone
    )

    return interactive_map_grid_cmp

#========================================================================================
# Function 26: Validate input format
#========================================================================================
def input_validation(year_ykr,vector_type,map_type,
                     transport_mode,mode_cmp,grid_cmp,
                     interactive_map_display
                    ):
    """
    Validate the input format
    """
    
    # Valdiate input format
    assert type(year_ykr) == list, "year_ykr should be a list"
    assert vector_type in ("shp", "gpkg"), "Only shp or gpkg"
    assert map_type in ("static", "interactive"), "static or interactive"
    assert transport_mode in CODE_DICTIONARY.keys(), "Unsupported transport mode"

    # Validate input format of each item in year_ykr
    for i in year_ykr:
        assert i[:4] in ("2013","2015","2018"), "2013, 2015 or 2018 only"
        assert i[4:5] == "_", "Year should be seperated from YKR ID with (_)"
        assert len(i[5:]) == 7, "YKR ID should have 7 digits"
        assert len(i) == 12, "Each item should have 12 characters"

    # Validate if mode_cmp is provided
    if mode_cmp:
        assert type(mode_cmp) == list, "mode_cmp should be a list"
        assert len(mode_cmp) == 2, "Two items only"
        # Validate input format of each item in mode_cmp
        for i in mode_cmp:
            assert i in CODE_DICTIONARY.keys(), "Unsupported transport mode"

    # Validate if grid_cmp is provided
    if grid_cmp:
        assert type(grid_cmp) == list, "grid_cmp should be a list"
        assert len(grid_cmp) == 2, "Two items only"
        assert grid_cmp[0][5:12] == grid_cmp[1][5:12], "YKR ID should be the same"
        assert not grid_cmp[0][:4] == grid_cmp[1][:4], "Assign different year"
        # Validate input format of each item in grid_cmp
        for i in grid_cmp:
            assert i[:12] in year_ykr, "File must appear in year_ykr"
            assert i[-1:] == "t", "for travel‑time difference only"
            assert i[:4] in ("2013","2015","2018"), "2013, 2015 or 2018 only"
            assert i[4:5] == "_" and i[12:13] == "_", "Separated by (_)"
    
    # Validate if interactive_map_display is provided
    if interactive_map_display:
        assert interactive_map_display in ("Yes", "No"), "Either Yes or No"

#========================================================================================
# Function 27: Check whether the specified transport mode has data for the given year.
#========================================================================================
def check_year(df,data,year,ykr_id):
    """
    Check whether the specified transport mode has data for the given year.

    Input:
    df: geodataframe
    data:column for plotting
    year: str
    ykr_id: str
    """

    # For files in three different years
    # Check whether the transport mode column exists or not
    if year == "2018":
        assert data in YEAR_CODE_DICTIONARY["2018"],\
        f"No {data} data in {year}_{ykr_id}" 
    if year == "2015":
        assert data in YEAR_CODE_DICTIONARY["2015"],\
        f"No {data} data in {year}_{ykr_id}"    
    if year == "2013":
        assert data in YEAR_CODE_DICTIONARY["2013"],\
        f"No {data} data in {year}_{ykr_id}" 
        
    # If the transport mode column exists
    # Check whether the transport_mode category contains any valid data
    # If the whole column contains invalid data, this tool wont plot the data
    # -1: no data; 0: at the destination row
    
    assert not df[data].isin([-1, 0]).all(),\
    f"No {data} data in {year}_{ykr_id}" 
    
#========================================================================================
# Function 28: Check the validity of the data
#========================================================================================
def valid_data_checking(grid_path,data_catergory):
    """
    Check whether there is any valid data in the specified transport mode catergory 
    for map visulaiztion. Raise a assertion if the data has no any valid data.

    Input:
    - grid_path: Path 
        - File path for travel time data spatial layer
    - data_catergory: columns for plotting
    """

    # Extract the YKR ID destination and year from the filename
    ykr_id = split_path_ykr(grid_path)
    year = split_path_year(grid_path)

    # Retrieve the travel time data file
    df = geopandas.read_file(grid_path)
        
    # Only travel time/distance columns remained
    df = df.drop(columns=["from_id","to_id", "YKR_ID", "x", "y","geometry"])

    # Make a list
    columns_name = list(df.columns)

    # Trigger an error if the whole travel time data file contains invalid data
    # -1: no data; 0: at the destination row
    # e.g 2013_6016691 has no any valid data
    assert not df[columns_name].isin([-1, 0]).all(axis=0).all(),\
    f"This dataset has no valid values. Delete {grid_path.stem} in year_ykr."

    # If the data file at least contains some data 
    # Check if there is any data in the specified catergory
    # e.g some column 2015_6016692 has no data
    # If user specifiy those no-value column for plotting
    # Raise an assertion
    for catergory in data_catergory:
        check_year(df=df,data=catergory,year=year,ykr_id=ykr_id)
        
#========================================================================================
# Function 29: Access Viz tool
#========================================================================================
def access_viz (
    year_ykr,
    vector_type,vector_output_folder,
    map_type,map_output_folder,
    transport_mode,mode_cmp,grid_cmp,
    interactive_map_display
):
    """
    Visualize and compare the travel time data in the Helsinki Region

    Input:
    - year_ykr: list
        - List of year_YKR ID, formatted as "2018_5785640"
    - vector_type: str
        - Format: either "shp" for Shapefile or "gpkg" for Geopackage
    - vector_output_folder: str
        - Folder path where output files will be saved
    - map_type: str
        - Format: "static" or "interactive"
    - map_output_folder: str
        - Folder path where output files will be saved
    - transport_mode: str
        - Transport mode of interest
    - mode_cmp: list 
        - List of exactly two different travel time OR two different distance categories
        - Format: ["bike_s_t","pt_m_t"] 
    - grid_cmp: list
        - List of two items: year_YKR ID_travel time category
        - Format: ["2018_6016691_bike_s_t","2015_6016691_pt_m_t"]
    - interactive_map_display: str
        - static: None
        - interactive: "Yes" or "No"
    """
    
    # Validate all input format
    input_validation(year_ykr,vector_type,map_type,
                     transport_mode,mode_cmp,grid_cmp,
                     interactive_map_display
                    )

    # Create the spatial file for the selected YKR ID
    grid_path=join_travel_time_table(
        year_ykr=year_ykr,
        vector_type=vector_type,
        vector_output_folder=vector_output_folder
    )

    # Write the file path to txt file
    write_to_txt(grid_path)

    # Save the grid_path in a dictionary
    grid_lookup = {
        path.stem[:12]: path
        for path in grid_path
    }
    
    # Define a list to store all maps
    all_map = []

    # For every travel time files:
    for path in grid_path:

        # Validate if the specified transport_mode has any data for plotting
        valid_data_checking(grid_path=path,data_catergory=[transport_mode])
        
        # Collect the key componenet of the filename
        label = CODE_DICTIONARY.get(transport_mode)
        ykr_id = split_path_ykr(path)
        year = split_path_year(path)
        
        # Plot
        if map_type == "static":
            static_map_travel_time = plot_static_map_travel_time(
                grid_path=path,
                map_type=map_type,
                transport_mode=transport_mode
            )
    
            # Append to the list
            all_map.append({
                "map": static_map_travel_time,
                "file_name":f"Travel Time ({year}) Map ({label}) To YKR ID {ykr_id}"
            })
                           
            # if mode_cmp is provided
            if mode_cmp: 
                
                # Validate if the specified transport_mode has any data for plotting
                merged = mode_cmp + [transport_mode]
                valid_data_checking(grid_path=path,data_catergory=merged)

                # Collect the key componenet of the filename
                mode_1, mode_2 = mode_cmp
                                
                # Plot
                static_map_mode_cmp = plot_static_map_mode_cmp(
                    grid_path=path,
                    map_type=map_type,
                    mode_cmp=mode_cmp
                )

                # Append to the list
                all_map.append({
                    "map": static_map_mode_cmp,
                    "file_name":f"{year}_{mode_1}_VS_{mode_2} To YKR ID {ykr_id}"
                })
        
        # Plot interactive map
        if map_type == "interactive":

            # Plot
            interactive_map_travel_time = plot_interactive_map_travel_time(
                grid_path=path,
                map_type=map_type,
                transport_mode=transport_mode
            )
    
            # Append to the list
            all_map.append({
                "map": interactive_map_travel_time,
                "file_name":f"Travel Time ({year}) Map ({label}) To YKR ID {ykr_id}"
            })
                           
            # if mode_cmp is provided
            if mode_cmp: 

                # Validate if the specified transport_mode has any data for plotting
                merged = mode_cmp + [transport_mode]
                valid_data_checking(grid_path=path,data_catergory=merged)

                # Collect the key componenet of the filename
                mode_1, mode_2 = mode_cmp
                
                # Plot
                interactive_map_mode_cmp = plot_interactive_map_mode_cmp(
                    grid_path=path,
                    map_type=map_type,
                    mode_cmp=mode_cmp 
                )

                # Append to the list
                all_map.append({
                    "map": interactive_map_mode_cmp,
                    "file_name":f"{year}_{mode_1}_VS_{mode_2} To YKR ID {ykr_id}"
                })
   
    # if grid_cmp is provided
    if grid_cmp: 

        # Collect the key componenet of the filename
        grid_1, grid_2 = grid_cmp
        
        # Calculate the travel time difference between two files
        grid_travel_data_cmp,grid_path_cmp = compare_grids_t2t(
            grid_cmp=grid_cmp,
            vector_output_folder=vector_output_folder,
            vector_type=vector_type,
            grid_lookup=grid_lookup
        )
        
        # Plot static map
        if map_type == "static":
            static_map_grid_cmp = plot_static_map_grid_cmp(
                grid_path_cmp=grid_path_cmp,
                grid_cmp=grid_cmp,
                map_type=map_type,
                vector_type=vector_type,
                vector_output_folder=vector_output_folder,
                grid_lookup=grid_lookup
            )

            # Append to the list
            all_map.append({
            "map": static_map_grid_cmp,
            "file_name":f"{grid_1}_VS_{grid_2}"
            })

        # Plot interactive map
        if map_type == "interactive":
            interactive_map_grid_cmp = plot_interactive_map_grid_cmp(
                grid_path_cmp=grid_path_cmp,
                grid_cmp=grid_cmp,
                map_type=map_type,
                vector_type=vector_type,
                vector_output_folder=vector_output_folder,
                grid_lookup=grid_lookup
            )

            # Append to the list
            all_map.append({
            "map": interactive_map_grid_cmp,
            "file_name":f"{grid_1}_VS_{grid_2}"
            })

    # Export
    for item in all_map:
        if map_type == "static":
            output_path = map_output_folder / f"{item['file_name']}.png"
            fig = item["map"].get_figure()
            fig.savefig(output_path, dpi=300)
            print(f"Saved as {output_path.name}")
            
        if map_type == "interactive":
            output_path = map_output_folder / f"{item['file_name']}.html"
            item["map"].save(str(output_path))
            print(f"Saved as {output_path.name}")

    # Display interactive maps
    if map_type == "interactive" and interactive_map_display == "Yes":
        for item in all_map:
            display(item["map"])
    
    return all_map


#========================================================================================
# Function 30: Access Viz tool_shortest possible path
#========================================================================================
def shortest_path(transport,origin,destination,map_output_folder):
    """
    Calculate and visualize the shortest possible path in Helsinki Region
    
    Input:
    - transport: str
        - "bike", "drive", "walk"
    - origin: str
    - destination: str
    - map_output_folder: str
        - Folder path where output files will be saved

    Output:
    - static map
    """

    # Define place name
    PLACE_NAME = "Helsinki, Finland"

    # Get a area with 200m within Helsinki, Finland
    # Get the area polygon once
    place_polygon = ox.geocode_to_gdf(PLACE_NAME)
    # Reproject to metric CRS for buffering
    place_polygon = place_polygon.to_crs("EPSG:3067")
    # Buffer by 200 meters
    place_polygon["geometry"] = place_polygon.buffer(200)
    # Reproject back to WGS84 for OSMnx
    place_polygon = place_polygon.to_crs("EPSG:4326")
    polygon = place_polygon.at[0, "geometry"]

    # Inform the users about the progress
    print(f"Downloading {transport} network…")
    
    # Retrieve the network graph
    graph = ox.graph_from_polygon(
        polygon,
        network_type=transport
    )

    # Transform the graph
    graph = ox.project_graph(graph,to_crs="EPSG:3067") 

    # Extract reprojected edges
    nodes, edges = ox.graph_to_gdfs(graph)

    # Define origin
    origin = (
        ox.geocode_to_gdf(origin)  # fetch geolocation
        .to_crs(edges.crs)  # transform to UTM
        .at[0, "geometry"]  # pick geometry of first row
        .centroid  # use the centre point
    )

    # Define destination
    destination = (
        ox.geocode_to_gdf(destination)
        .to_crs(edges.crs)
        .at[0, "geometry"]
        .centroid)

    # Analyze the shortest path
    origin_node_id = ox.nearest_nodes(graph, origin.x, origin.y)
    destination_node_id = ox.nearest_nodes(graph, destination.x, destination.y)
    route = ox.shortest_path(graph, origin_node_id, destination_node_id)

    # Convert route to GeoDataFrame
    route_gdf = ox.routing.route_to_gdf(graph, route)

    # Convert origin and destination to GeoDataFrame
    point_gdf = geopandas.GeoDataFrame(
        geometry=[
            Point(origin.x, origin.y),
            Point(destination.x, destination.y)
        ],
        crs=route_gdf.crs
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot route 
    route_gdf.plot(
        ax=ax,
        linewidth=4,
        color="red",
        zorder=3
    )

    # Plot origin & destination
    ax.scatter(
        [origin.x, destination.x],
        [origin.y, destination.y],
        c=["green", "blue"],
        s=100,
        zorder=4,
        edgecolor="white"
    )

    # Zoom to map extent covering the origin, route and destination
    # Combine route + origin/destination geodataframe
    combined_gdf = pd.concat(
        [route_gdf, point_gdf],
        ignore_index=True
    )

    # Zoom to extent
    min_x, min_y, max_x, max_y = combined_gdf.total_bounds

    # Create map extent buffer
    buffer = 300 
    ax.set_xlim(min_x - buffer, max_x + buffer)
    ax.set_ylim(min_y - buffer, max_y + buffer)

    # Add Basemap
    contextily.add_basemap(
        ax,
        source=contextily.providers.OpenStreetMap.Mapnik,
        alpha=1,
        crs=route_gdf.crs,  # EPSG:3067
        attribution=(
            "Travel time data: Digital Geography Lab;\n"
            "Basemap: OpenStreetMap contributors; CRS: EPSG:3857"
        )
    )

    # Add figure title
    ax.set_title(f"Shortest distance between {origin}(green) and {destination}(blue)")

    # Export
    output_path = map_output_folder / f"shortest_path_calculation.png"
    fig.savefig(output_path, dpi=300)
    print(f"Saved as {output_path.name}")

    plt.show()

    



### Structure of this repository:
This repository consists of a [Jupyter notebook](final-assignment.ipynb) and a [function.py file](function.py). Inside the function.py file, Function 1 creates an interactive YKR grid index map, and Function 29 serves as the Access Viz tool. Function 30, which present the additional function of this tool, uses to visualize shortest path routes (walking, cycling, and/or driving) using OpenStreetMap data. All remaining functions are used within the Access Viz tool for data input, processing, and map visualization.

### Input data:
The travel time input data is derived from the Helsinki Region Travel Time Matrix (2013, 2015, and 2018), which reports travel times and distances between all 250 m × 250 m grid‑cell centroids (n = 13,231) in the Helsinki Capital Region for walking, cycling, public transport, and car travel. A small sample of the input data is saved inside [the data folder/](data/) according to different year. Please note that the filepath must not be changed. If users want to include additional datasets, they can download them from [here](https://blogs.helsinki.fi/accessibility/helsinki-region-travel-time-matrix/). Simply place the downloaded files into the designated folder. The subfolder structure should be the same as the sample. No need to change anything of the downloaded file.

The MetropAccess-YKR-grid.shp and HSL Transif Zone.gpkg file are also saved inside data folder [data/](data/). Please note that the filepath must not be changed.

Data from OpenStreetMap data is used to calculate the shortest path route. 

### Analysis steps:
The first step in the Access Viz tool is to ensure that all input parameters follow the required format defined in the code. All assertion‑related checks are implemented in Function 28, which raises an assertion error if any parameter is invalid. If the travel‑time file contains no data, or if the selected transport mode has no data within that file, an assertion error is raised. Refer to [Jupyter notebook](final-assignment.ipynb) for more guidance.

Once the input parameters are validated, the tool retrieves the travel‑time data file from the [the data folder/](data/) and joins it with the `YKR grid` to create a GeoDataFrame. The `vector_type` and `vector_output_folder` allows the users to export the file as a `.shp` or `.gpkg` file in a defined folder. The tool then generate the map of the specified `transport_mode` in each item of `year_ykr`. The map can be either a static or interactive map based on `map_type`. The user can define the map output folder in `map_output_folder`.

The tool can run without `mode_cmp` or `grid_cmp`. If `mode_cmp` is provided, the tool computes the travel‑time or distance differences and produces the corresponding static or interactive map. If `grid_cmp` is provided, the tool calculates the travel‑time differences between two files, creates a new DataFrame to store these values, and allows exporting it as `.shp` or `.gpkg`. A static or interactive map is then generated according to the `map_type`. The user can define the map output folder in `map_output_folder`. Please note that any travel‑time file used in `grid_cmp` must also be included in the `year_ykr`parameter since travel time data file search is based on the `year_ykr`parameter ONLY.

`interactive_map_display` allows the user to decide whether the interactive map should be shown directly in the Jupyter notebook. Displaying too many interactive maps can cause the notebook to exceed its storage limits, especially on the Noppe platform. Static maps are always displayed in the Jupyter notebook. If `interactive_map_display` is set to `None`, the interactive map will only be saved. 

The function.py file contains plotting functions designed separately for static and interactive maps. The overall logic is to consolidate all shared map elements into a base function and then append map‑specific components as needed.

The tool output spatial file (.shp or .gpkg) and map (.png or .html) accordingly. 

- txt file of the travel time filepath

For each item in the `year_ykr`parameter:
- travel time data.shp OR .gpkg
- static map OR interactive map showing the travel time data

If `mode_cmp` is provided:
For each item in the `year_ykr`parameter:
- static map OR interactive map showing the travel time/distance difference

If `grid_cmp` is provided:
- travel time difference between two files.shp OR .gpkg
- static map OR interactive map showing the travel time difference between two files

For the additional function of this tool, the shortest possible path route is calculated using the data from the OpenStreetMap using OSMnx package

### Results:
The output map comprises of the travel time data, annotated HSL zone and the destination marker. Therefore, users can easily visualize the spatial relationship of different area based on the travel time data file. In addition, the additional function of this tool allow calculating and visualizing the shortest possible path route.

### References:
- Travel time data: Digital Geography Lab
- Helsinki Region Transport tariff zones: Helsinki Region Infoshare
- Basemap: OpenStreetMap contributors
        

# Vibe-Running

The **Vibe-Running** is a Flask-based web application that visualizes running race data on interactive maps. It processes race data from an Excel file, geocodes locations, and generates visualizations, including single SVG maps, multiple SVG frames, or an MKV video showing race locations and counts over time. 

## Goal of the Program
Vibe-Running aims to help runners visualize their race history geographically and temporally. It takes race data and produces:
- **Single SVG Map**: A world map showing all races up to the latest race date, with a heatmap, markers colored by race type (e.g., Marathon, 50km, 100km, etc.), and a legend listing race counts and total.
- **Multiple SVG Frames**: A series of SVG maps (e.g., `frame_001.svg` to `frame_nnn.svg`), each showing races up to a specific year-month, ideal for tracking progress over time.
- **MKV Video**: A 60-second video animating the SVG frames, showing the progression of races across months.

The visualizations include a world map with Cartopy features (land, coastlines, borders, states), a heatmap of race density, and a legend detailing race types (e.g., “Marathon: 200”, “Total: 333”). This project was developed to combine my passion for long-distance running and interests in Generative AI...

## Vibe-Coding with GenAI
This program was coded in **less than a day** using a **vibe-coding** approach, an iterative and collaborative development process with significant assistance from **Grok**, **Perplexity.ai** and **ChatGPT**. Grok and Perplexity.ai helped develop the core framework of the program, while ChatGPT helped refine the codebase, implement features (e.g., legend formatting, endpoint logic), debug issues, and optimize visualizations. The vibe-coding process involved back-and-forth exploration to align the program with the desired functionality, blending human creativity with AI-driven insights.

## Dependencies and Installation
The **Vibe-Running** requires Python 3.11 and several Python packages, along with external tools (FFmpeg and Graphviz). Below are the dependencies and Windows-specific installation steps.

### Dependencies
- **(non-exhaustive) Python Packages**:
  - `pandas`: Data processing for Excel files and DataFrames.
  - `matplotlib`: Plotting maps and visualizations.
  - `cartopy`: Geospatial mapping with world map features.
  - `scipy`: Gaussian KDE for heatmap generation.
  - `geopy`: Geocoding race locations.
  - `flask`: Web API framework.
  - `cairosvg`: SVG-to-PNG conversion for MKV video.
  - `psutil`: Memory usage monitoring.
  - `ffmpeg-python`: FFmpeg integration for video generation.
- **External Tools**:
  - **FFmpeg**: Video encoding for MKV output.
  - **Graphviz**: Required by `cairosvg` for rendering (bundled with GIMP).

### Installation Steps (Windows)
1. **Install Python 3.11**:
   - Download and install Python 3.11 from [python.org](https://www.python.org/downloads/release/python-3110/).
   - Verify:
     ```bash
     python --version
     ```
     Output: `Python 3.11.x`

2. **Install Python Packages**:
   - Use `pip` to install dependencies:
     ```bash
     pip install pandas matplotlib cartopy scipy geopy flask cairosvg psutil ffmpeg-python
     ```

3. **Install FFmpeg**:
   - Install via Chocolatey (package manager):
     ```bash
     choco install ffmpeg
     ```
   - Verify (should point to `C:\ProgramData\chocolatey\bin\ffmpeg.exe`):
     ```bash
     ffmpeg -version
     ```
   - If Chocolatey is not installed:
     ```bash
     choco install chocolatey
     ```

4. **Install Graphviz (for Cairo)**:
   - Install GIMP, which bundles `libcairo-2.dll` and Graphviz dependencies:
     - Download from [gimp.org](https://www.gimp.org/downloads/) and install to `C:\Program Files\GIMP 2`.
   - Add GIMP’s bin directory to PATH:
     ```bash
     setx PATH "%PATH%;C:\Program Files\GIMP 2\bin"
     ```
   - Verify Cairo:
     ```python
     python -c "from cairosvg import svg2png; svg2png(url='outputs/frame_001.svg', write_to='test.png')"
     ```
     - If this fails, ensure `C:\Program Files\GIMP 2\bin` contains `libcairo-2.dll`, `zlib1.dll`, `libpng16-16.dll`.

5. **Clone the Repository**:
   - Clone the project:
     ```bash
     git clone https://github.com/jgreboul/vibe-running.git
     cd vibe-running
     ```

### Notes
- Ensure `C:\ProgramData\chocolatey\bin\ffmpeg.exe` is accessible (used by `matplotlib.rcParams['animation.ffmpeg_path']`).
- If OneDrive syncs the project folder (`<Your Code Folder>\vibe-running`), pause syncing during runs to avoid file locking.
- All commands assume a Windows Command Prompt; adjust for PowerShell if needed (e.g., use `.\` for executables).

## Endpoints

**Vibe-Running** provides three endpoints to upload race data, geocode locations, and generate visualizations. All endpoints are POST requests and run on `http://localhost:5000` by default.

### 1. `/upload_run`
- **Purpose**: Uploads an Excel file containing race data and stores it in memory.
- **Input**:
  - Form-data file upload with key `file`.
  - Excel file (`.xlsx`) with columns: `ID`, `Event`, `ExcelDate` (Excel serial date or timestamp), `City`, `State`, `Country`, `Distance` (e.g., “Marathon”, “50km”).
- **Output**:
  - JSON: `{"status": "success", "message": "File uploaded and processed, data stored in memory"}` (HTTP 200).
  - Errors: `{"error": "..."}` (HTTP 400/500, e.g., “No file provided”, “File must be an Excel (.xlsx) file”).
- **Example**:
  ```bash
  curl -X POST http://localhost:5000/upload_run -F "file=@races.xlsx" -o upload_response.json
  ```
  - Input file (`races.xlsx`): ~400 rows, with races like `ID=1, Event="Boston Marathon", ExcelDate=44734, City="Boston", State="MA", Country="USA", Distance="Marathon"`.
  - Output (`upload_response.json`): `{"status": "success", "message": "File uploaded and processed, data stored in memory"}`.
- **Notes**:
  - Converts `ExcelDate` to `YYYY-MM-DD` (e.g., Excel serial 44734 → “2025-05-06”).
  - Initializes `Latitude` and `Longitude` as `None`.

### 2. `/set_location`
- **Purpose**: Geocodes race locations, updates coordinates in memory, and generates a GeoJSON for visualization.
- **Input**:
  - JSON body with `geocoded_locations` (optional):
    ```json
    {
      "geocoded_locations": [
        {"location": "Boston, MA, USA", "latitude": 42.3601, "longitude": -71.0589},
        ...
      ]
    }
    ```
  - If no locations provided, uses Nominatim to geocode unique `City, State, Country` combinations.
- **Output**:
  - JSON: `{"status": "success", "message": "Locations geocoded and GeoJSON stored in memory", "geocoded_locations": [...], "warnings": [...]}` (HTTP 200).
    - `geocoded_locations`: List of all locations with coordinates.
    - `warnings`: Notes if input exceeds 250 locations (processes first 250).
  - Errors: `{"error": "..."}` (HTTP 400/500, e.g., “No race data found”).
- **Example**:
  ```bash
  curl -X POST http://localhost:5000/set_location -H "Content-Type: application/json" -d "{\"geocoded_locations\": [{\"location\": \"Boston, MA, USA\", \"latitude\": 42.3601, \"longitude\": -71.0589}]}" -o location_response.json
  ```
  - Output (`location_response.json`): Includes all unique locations (e.g., ~100 from 400 races), with coordinates either from input or Nominatim.
- **Notes**:
  - Caches geocoding results to avoid repeated API calls.
  - Generates GeoJSON with features for each race, including popup data (e.g., “Boston Marathon<br>Date: 2025-05-06<br>Distance: Marathon”).
  - Requires `/upload_run` to be called first.

### 3. `/generate_map`
- **Purpose**: Generates visualizations (SVG map, SVG frames, or MKV video) of race locations with heatmaps, markers, and legends.
- **Input**:
  - Query parameter: `format` (required, one of `svg`, `svgs`, `mkv`) and optional `duration` (default 60 seconds for `mkv`).
  - Alternative: JSON body:
    ```json
    {"format": "mkv", "duration": 60}
    ```
- **Output**:
  - **format=svg**:
    - JSON: `{"message": "Generated SVG in outputs folder"}` (HTTP 200).
    - Visuals: World map, title “Races up to Month Day, Year” (e.g., “Races up to September 23, 2025”), heatmap, markers (`s=50`, colored by race type), legend (bottom center, “Marathon: N”, ..., “Total: X”).
  - **format=svgs**:
    - JSON: `{"message": "Generated N SVG frames in outputs folder"}` (HTTP 200).
    - Files: `frame_001.svg` to `frame_N.svg` in `outputs` (~7.5–15 MB for 151 frames).
    - Visuals: One SVG per year-month (e.g., 151 frames), same style as `svg` but progressing by month.
  - **format=mkv**:
    - JSON: `{"message": "Generated MKV video in outputs folder"}` (HTTP 200).
    - File: `output.mkv` in `outputs` (~10–50 MB, 60s, 24 FPS).
    - Visuals: Video animating SVG frames, with frame durations based on year-month spans.  

- **Notes**:
  - Requires `/upload_run` and `/set_location` to be called first.
  - `svg`: Shows all races up to the latest date (e.g., “September 23, 2025”).
  - `svgs`: Generates one frame per year-month, capped at 5000. Must be called before generating a `mkv`
  - `mkv`: Converts SVGs to PNGs (via `cairosvg`), then encodes to MKV (via FFmpeg), ~20–40s runtime.
  - Legend format: “Marathon: N”, “50km: M”, ..., “Total: X” (all 9 race types, including `0` counts), bottom center, 3 columns.
  - Heatmap: Red-orange, shown if ≥2 unique coordinates with sufficient variance.

## Usage Example
1. Start the Flask server:
   ```bash
   python.exe main.py > log.txt
   ```
2. Upload race data:
   ```bash
   curl -X POST http://localhost:5000/upload_run -F "file=@races.xlsx" -o upload_response.json
   ```
3. Geocode locations:
   ```bash
   curl -X POST http://localhost:5000/set_location -H "Content-Type: application/json" -d "{\"geocoded_locations\": []}" -o location_response.json
   ```
4. Generate visualizations:
   - SVG:
     ```bash
     curl -X POST http://localhost:5000/generate_map?format=svg -o svg_response.json
     ```
   - SVGs:
     ```bash
     curl -X POST http://localhost:5000/generate_map?format=svgs -o svgs_response.json
     ```
   - MKV:
     ```bash
     curl -X POST http://localhost:5000/generate_map?format=mkv&duration=30 -o mkv_response.json
     ```

## Notes
- **Logs**: Check `log.txt` for debugging (e.g., heatmap warnings, Cairo errors).
- **Performance**: SVG (~1–2s), SVGs (~45–75s for 151 frames), MKV (~20–40s).
- **Outputs**: Stored in `outputs` (`frame_latest.svg`, `frame_*.svg`, `output.mkv`). Clear before re-running:
  ```bash
  del outputs\frame_*.svg outputs\output.mkv
  ```
- **Data**: Ensure `races.xlsx` has valid dates and locations. Invalid data may cause heatmap or geocoding errors (logged in `log.txt`).

# Contact Information
Enjoy Vibe-Coding!
And for more insights, subscribe to my Youtube Channel: https://www.youtube.com/user/jgreboul 
Thank you, 
Jean-Gael (Jay) Reboul
jgreboul@gmail.com


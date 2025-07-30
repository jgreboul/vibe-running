import pandas as pd
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import gc
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend
matplotlib.rcParams['animation.ffmpeg_path'] = 'C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe'


app = Flask(__name__)
pp = Flask(__name__)

# Initialize geocoder with user agent
geolocator = Nominatim(user_agent="vibe-running")

# Global in-memory storage for race data and GeoJSON
race_data = None
race_geojson = None

# Cache for geocoding results
geocode_cache = {}
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to convert Excel serial date or Timestamp to standard date
def excel_to_date(excel_date):
    try:
        if isinstance(excel_date, pd.Timestamp):
            return excel_date.strftime("%Y-%m-%d")
        if isinstance(excel_date, (int, float)):
            base_date = datetime(1900, 1, 1)
            standard_date = base_date + timedelta(days=int(excel_date) - 2)  # Adjust for Excel leap year bug
            return standard_date.strftime("%Y-%m-%d")
        print(f"Invalid date format: {excel_date}")
        return None
    except Exception as e:
        print(f"Error converting date {excel_date}: {e}")
        return None

# Function to geocode a location with caching
def geocode_location(city, state, country):
    location_key = f"{city}, {state}, {country}"
    if location_key in geocode_cache:
        return geocode_cache[location_key]
    
    try:
        location = geolocator.geocode(location_key, timeout=10)
        if location:
            coords = [location.latitude, location.longitude]
            geocode_cache[location_key] = coords
            return coords
        else:
            print(f"Geocoding failed for {location_key}")
            return [None, None]
    except Exception as e:
        print(f"Error geocoding {location_key}: {e}")
        return [None, None]

# Endpoint 1: Upload Run
@app.route("/upload_run", methods=["POST"])
def upload_run():
    global race_data
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    if not file.filename.endswith(".xlsx"):
        return jsonify({"error": "File must be an Excel (.xlsx) file"}), 400
    
    try:
        # Read Excel file
        df = pd.read_excel(file, header=None)
        df.columns = ["ID", "Event", "ExcelDate", "City", "State", "Country", "Distance"]
        
        # Convert dates
        df["Date"] = df["ExcelDate"].apply(excel_to_date)
        
        # Initialize coordinates
        df["Latitude"] = None
        df["Longitude"] = None
        
        # Store in memory
        race_data = df
        return jsonify({
            "status": "success",
            "message": "File uploaded and processed, data stored in memory"
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

# Endpoint 2: Set Location
@app.route("/set_location", methods=["POST"])
def set_location():
    global race_data, race_geojson
    if race_data is None:
        return jsonify({"error": "No race data found. Please upload runs first."}), 400
    
    try:
        # Parse JSON payload
        data = request.get_json(silent=True)
        provided_locations = []
        warnings = []
        
        if data and "geocoded_locations" in data and data["geocoded_locations"]:
            geocoded_locations = data["geocoded_locations"]
            if len(geocoded_locations) > 250:
                warnings.append(f"Input contains {len(geocoded_locations)} locations, exceeding 250. Processing only the first 250.")
                geocoded_locations = geocoded_locations[:250]
            
            provided_locations = []
            for loc in geocoded_locations:
                if not all(key in loc for key in ["location", "latitude", "longitude"]):
                    print(f"Invalid location entry: {loc}")
                    continue
                try:
                    # Parse location string (e.g., "Lynnwood, WA, USA")
                    location = loc["location"]
                    parts = [p.strip() for p in location.split(",")]
                    if len(parts) != 3:
                        print(f"Invalid location format: {location}")
                        continue
                    city, state, country = parts
                    latitude = float(loc["latitude"])
                    longitude = float(loc["longitude"])
                    provided_locations.append({
                        "city": city,
                        "state": state,
                        "country": country,
                        "latitude": latitude,
                        "longitude": longitude
                    })
                except (ValueError, KeyError) as e:
                    print(f"Error parsing location {loc}: {e}")
                    continue
        
        # Create a mapping of all coordinates
        location_coords = {}
        
        # Process provided locations first
        for loc in provided_locations:
            location_key = f"{loc['city']}, {loc['state']}, {loc['country']}"
            location_coords[location_key] = [loc["latitude"], loc["longitude"]]
        
        # Identify unique locations in race_data
        unique_locations = race_data[["City", "State", "Country"]].drop_duplicates()
        
        # Geocode remaining unique locations not provided in JSON
        if not provided_locations or len(provided_locations) < len(unique_locations):
            for _, row in unique_locations.iterrows():
                city, state, country = row["City"], row["State"], row["Country"]
                location_key = f"{city}, {state}, {country}"
                if location_key not in location_coords:
                    coords = geocode_location(city, state, country)
                    location_coords[location_key] = coords
        
        # Update race_data with coordinates
        def assign_coords(row):
            location_key = f"{row['City']}, {row['State']}, {row['Country']}"
            return pd.Series(location_coords.get(location_key, [None, None]))
        
        race_data[["Latitude", "Longitude"]] = race_data.apply(assign_coords, axis=1)
        
        # Generate GeoJSON
        features = []
        for _, row in race_data.dropna(subset=["Latitude", "Longitude", "Date"]).iterrows():
            color = {
                "Marathon": "red",
                "50km": "blue",
                "Ultra": "purple",
                "50 Miler": "green",
                "100km": "darkgreen",
                "120km": "darkblue",
                "100 Miler": "orange",
                "150 Miler": "darkorange",
                "200 Miler": "black"
            }.get(row["Distance"], "black")
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["Longitude"], row["Latitude"]],
                },
                "properties": {
                    "time": row["Date"],
                    "popup": f"{row['Event']}<br>Date: {row['Date']}<br>Distance: {row['Distance']}<br>Location: {row['City']}, {row['State']}, {row['Country']}",
                    "icon": "circle",
                    "style": {
                        "fillColor": color,
                        "fillOpacity": 0.8,
                        "stroke": True,
                        "color": "black",
                        "weight": 1,
                    },
                },
            }
            features.append(feature)
        
        race_geojson = {"type": "FeatureCollection", "features": features}
        
        # Prepare response with geocoded locations
        geocoded_locations = [
            {
                "location": location_key,
                "latitude": coords[0] if coords[0] is not None else None,
                "longitude": coords[1] if coords[1] is not None else None
            }
            for location_key, coords in location_coords.items()
        ]
        
        return jsonify({
            "status": "success",
            "message": "Locations geocoded and GeoJSON stored in memory",
            "geocoded_locations": geocoded_locations,
            "warnings": warnings
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error geocoding locations: {str(e)}"}), 500

def generate_svg_map(df):
    import logging
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from scipy.stats import gaussian_kde
    from io import BytesIO

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        df = df.dropna(subset=["Latitude", "Longitude", "Date"])
        if df.empty:
            raise ValueError("No valid coordinates available for mapping")
        
        # Convert dates and get latest full date
        df["Date"] = pd.to_datetime(df["Date"])
        latest_date = df["Date"].max().strftime("%B %d, %Y")
        frame_df = df[df["Date"] <= df["Date"].max()]
        
        # Set fixed world map extent
        lon_min, lon_max = -180, 180
        lat_min, lat_max = -90, 90
        
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.STATES)

        # Set title with latest full date
        ax.set_title(f"Races up to {latest_date}")
        
        # Generate density heatmap
        x = frame_df["Longitude"].values
        y = frame_df["Latitude"].values
        if len(x) > 1:
            coords = np.vstack([x, y]).T
            unique_coords = np.unique(coords, axis=0)
            if len(unique_coords) > 1:
                x_std = np.std(x)
                y_std = np.std(y)
                min_std = 1e-6
                if x_std > min_std and y_std > min_std:
                    logger.info(f"Computing KDE heatmap, {len(x)} points, {len(unique_coords)} unique, x_std={x_std:.6f}, y_std={y_std:.6f}")
                    x_grid, y_grid = np.mgrid[lon_min:lon_max:50j, lat_min:lat_max:50j]
                    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
                    values = np.vstack([x, y])
                    kernel = gaussian_kde(values)
                    z = np.reshape(kernel(positions).T, x_grid.shape)
                    ax.contourf(
                        x_grid, y_grid, z,
                        cmap="hot", alpha=0.5, transform=ccrs.PlateCarree()
                    )
                else:
                    logger.warning(f"Skipping KDE heatmap: insufficient variance (x_std={x_std:.6f}, y_std={y_std:.6f})")
            else:
                logger.warning(f"Skipping KDE heatmap: only {len(unique_coords)} unique coordinate(s)")
        else:
            logger.warning(f"Skipping KDE heatmap: insufficient data ({len(x)} points)")
        
        # Plot scatters and collect counts for legend (include all race types)
        distance_colors = [
            ("Marathon", "red"),
            ("50km", "blue"),
            ("Ultra", "purple"),
            ("50 Miler", "green"),
            ("100km", "darkgreen"),
            ("120km", "darkblue"),
            ("100 Miler", "orange"),
            ("150 Miler", "darkorange"),
            ("200 Miler", "black")
        ]

        legend_labels = []
        handles = []
        total_races = 0
        for distance, color in distance_colors:
            subset = frame_df[frame_df["Distance"] == distance]
            count = subset.shape[0]
            total_races += count
            # Always include in legend, even if count == 0
            if count > 0:  # Only plot scatter if races exist
                scatter = ax.scatter(
                    subset["Longitude"], subset["Latitude"],
                    color=color, s=30, label=f"{distance}: {count}",
                    transform=ccrs.PlateCarree(), edgecolors="black", linewidth=0.5
                )
                handles.append(scatter)
            else:
                # Use a dummy scatter for legend with no data
                scatter = plt.scatter([], [], color=color, s=30, label=f"{distance}: {count}",
                                        edgecolors="black", linewidth=0.5)
                handles.append(scatter)
            legend_labels.append(f"{distance}: {count}")
        
        # Add total races to legend
        handles.append(plt.scatter([], [], c='none', label=f"Total: {total_races}"))
        legend_labels.append(f"Total: {total_races}")
        
        # Place legend at the bottom center
        ax.legend(
            handles=handles,
            labels=legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=3,
            frameon=True
        )
        
        # Create outputs folder
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Save frame
        output_path = os.path.join(output_dir, f"frame_latest.svg")
        fig.savefig(output_path, format='svg')
        plt.close(fig)
        
        # Clear memory
        gc.collect()

        logger.info("SVG generation complete")
        return {"message": f"Generated SVG in outputs folder"}

    except Exception as e:
        logger.error(f"Error generating SVG: {str(e)}")
        raise Exception(f"Error generating SVG: {str(e)}")

# Function to generate SVG frames in outputs folder
def generate_svgs(df):
    import logging
    import time
    import gc
    import psutil
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from scipy.stats import gaussian_kde
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting SVGs generation")
        df = df.dropna(subset=["Latitude", "Longitude", "Date"])
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Group dates by year-month
        df["YearMonth"] = df["Date"].dt.to_period("M")
        dates = sorted(df["YearMonth"].unique())
        logger.info(f"Found {len(dates)} unique year-months")
        
        # Cap frames at 5000 (only limit; all year-months processed unless >5000)
        max_frames = 5000
        if len(dates) > max_frames:
            logger.warning(f"Too many year-months ({len(dates)}), sampling to {max_frames}")
            date_indices = np.linspace(0, len(dates) - 1, max_frames, dtype=int)
            dates = [dates[i] for i in date_indices]
        
        frame_count = len(dates)
        logger.info(f"Generating {frame_count} SVG frames")
        
        # Create outputs folder
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Set fixed world map extent
        lon_min, lon_max = -180, 180
        lat_min, lat_max = -90, 90
        
        # Cache grid for heatmap (world map)
        x_grid, y_grid = np.mgrid[lon_min:lon_max:50j, lat_min:lat_max:50j]
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        
        # Generate SVG frames
        for frame_idx in range(len(dates)):
            start_time = time.time()
            process_memory = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Rendering SVG frame {frame_idx + 1}/{frame_count}, Memory: {process_memory:.2f} MB")
            
            # Initialize figure per frame
            fig = plt.figure(figsize=(9.6, 5.4))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(cfeature.STATES)
            
            # Filter data up to current year-month
            current_ym = dates[frame_idx]
            ax.set_title(f"Races up to {str(current_ym)}")
            frame_df = df[df["YearMonth"] <= current_ym]
            
            # Generate density heatmap if sufficient unique points and variance
            x = frame_df["Longitude"].values
            y = frame_df["Latitude"].values
            if len(x) > 1:
                # Check for unique points
                coords = np.vstack([x, y]).T
                unique_coords = np.unique(coords, axis=0)
                if len(unique_coords) > 1:
                    # Check for sufficient variance
                    x_std = np.std(x)
                    y_std = np.std(y)
                    min_std = 1e-6  # Minimum standard deviation (degrees)
                    if x_std > min_std and y_std > min_std:
                        logger.info(f"Computing KDE heatmap for frame {frame_idx + 1}, {len(x)} points, {len(unique_coords)} unique, x_std={x_std:.6f}, y_std={y_std:.6f}")
                        try:
                            values = np.vstack([x, y])
                            kernel = gaussian_kde(values)
                            z = np.reshape(kernel(positions).T, x_grid.shape)
                            ax.contourf(
                                x_grid, y_grid, z,
                                cmap="hot", alpha=0.5, transform=ccrs.PlateCarree()
                            )
                        except np.linalg.LinAlgError as e:
                            logger.warning(f"Skipping KDE heatmap for frame {frame_idx + 1}: singular covariance matrix (LinAlgError: {str(e)})")
                    else:
                        logger.warning(f"Skipping KDE heatmap for frame {frame_idx + 1}: insufficient variance (x_std={x_std:.6f}, y_std={y_std:.6f})")
                else:
                    logger.warning(f"Skipping KDE heatmap for frame {frame_idx + 1}: only {len(unique_coords)} unique coordinate(s)")
            else:
                logger.warning(f"Skipping KDE heatmap for frame {frame_idx + 1}: insufficient data ({len(x)} points)")
            
            # Plot scatters and collect counts for legend (include all race types)
            distance_colors = [
                ("Marathon", "red"),
                ("50km", "blue"),
                ("Ultra", "purple"),
                ("50 Miler", "green"),
                ("100km", "darkgreen"),
                ("120km", "darkblue"),
                ("100 Miler", "orange"),
                ("150 Miler", "darkorange"),
                ("200 Miler", "black")
            ]
            legend_labels = []
            handles = []
            total_races = 0
            for distance, color in distance_colors:
                subset = frame_df[frame_df["Distance"] == distance]
                count = subset.shape[0]
                total_races += count
                # Always include in legend, even if count == 0
                if count > 0:  # Only plot scatter if races exist
                    scatter = ax.scatter(
                        subset["Longitude"], subset["Latitude"],
                        color=color, s=30, label=f"{distance}: {count}",
                        transform=ccrs.PlateCarree(), edgecolors="black", linewidth=0.5
                    )
                    handles.append(scatter)
                else:
                    # Use a dummy scatter for legend with no data
                    scatter = plt.scatter([], [], color=color, s=30, label=f"{distance}: {count}",
                                         edgecolors="black", linewidth=0.5)
                    handles.append(scatter)
                legend_labels.append(f"{distance}: {count}")
            
            # Add total races to legend
            handles.append(plt.scatter([], [], c='none', label=f"Total: {total_races}"))
            legend_labels.append(f"Total: {total_races}")
            
            # Place legend at the bottom center
            ax.legend(
                handles=handles,
                labels=legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.2),
                ncol=3,
                frameon=True
            )
            
            # Save frame
            output_path = os.path.join(output_dir, f"frame_{frame_idx + 1:03d}.svg")
            fig.savefig(output_path, format='svg')
            plt.close(fig)
            
            # Clear memory
            gc.collect()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Frame {frame_idx + 1} rendered in {elapsed_time:.2f}s")
        
        logger.info("SVGs generation complete")
        return {"message": f"Generated {frame_count} SVG frames in outputs folder"}
    except Exception as e:
        logger.error(f"Error generating SVGs: {str(e)}")
        raise Exception(f"Error generating SVGs: {str(e)}")

# Function to generate MKV video from SVG frames in outputs folder
def generate_mkv_map(df, duration=60):
    import logging
    import os
    import glob
    import subprocess
    import shutil
    import tempfile
    import numpy as np
    import pandas as pd
    from cairosvg import svg2png
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting MKV video generation")
        
        # Validate input DataFrame
        df = df.dropna(subset=["Date"])
        df["Date"] = pd.to_datetime(df["Date"])
        df["YearMonth"] = df["Date"].dt.to_period("M")
        dates = sorted(df["YearMonth"].unique())
        num_frames = len(dates)
        logger.info(f"Found {num_frames} unique year-months")
        
        # Cap frames at 5000 (consistent with generate_svgs)
        max_frames = 5000
        if num_frames > max_frames:
            logger.warning(f"Too many year-months ({num_frames}), sampling to {max_frames}")
            date_indices = np.linspace(0, num_frames - 1, max_frames, dtype=int)
            dates = [dates[i] for i in date_indices]
            num_frames = max_frames
        
        # Find SVG frames in outputs folder
        output_dir = "outputs"
        frame_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.svg")))
        if not frame_files:
            raise ValueError("No SVG frames found in outputs folder")
        if len(frame_files) != num_frames:
            logger.warning(f"Expected {num_frames} SVG frames, found {len(frame_files)}")
        
        # Calculate frame durations based on date spans
        date_times = [pd.to_datetime(str(ym)) for ym in dates]
        time_deltas = np.diff(date_times, append=date_times[-1] + pd.offsets.MonthEnd(1))
        time_deltas_days = [delta.total_seconds() / (24 * 3600) for delta in time_deltas]
        total_days = sum(time_deltas_days)
        frame_durations = [(days / total_days) * duration for days in time_deltas_days]
        
        # Enforce minimum duration (0.1s) and normalize to exact duration
        min_duration = 0.1
        frame_durations = [max(d, min_duration) for d in frame_durations]
        current_total = sum(frame_durations)
        frame_durations = [d * (duration / current_total) for d in frame_durations]
        
        logger.info(f"Calculated frame durations: min={min(frame_durations):.2f}s, max={max(frame_durations):.2f}s, total={sum(frame_durations):.2f}s")
        
        # Create temporary directory for PNGs
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Convert SVGs to PNGs
        for i, frame_file in enumerate(frame_files, 1):
            png_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
            logger.info(f"Converting SVG frame {i}/{num_frames} to PNG")
            svg2png(url=frame_file, write_to=png_path, output_width=960, output_height=540)
        
        # Create FFmpeg input file with frame paths and durations
        ffmpeg_input_file = os.path.join(temp_dir, "frames.txt")
        with open(ffmpeg_input_file, "w") as f:
            for i, duration in enumerate(frame_durations, 1):
                f.write(f"file 'frame_{i:03d}.png'\n")
                f.write(f"duration {duration}\n")
        
        # Generate MKV video using FFmpeg
        output_mkv = os.path.join(output_dir, "output.mkv")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", ffmpeg_input_file,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-r", "24", "-s", "960x540",
            output_mkv
        ]
        logger.info(f"Running FFmpeg to generate MKV: {' '.join(ffmpeg_cmd)}")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        logger.info(f"MKV video generated at {output_mkv}")
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary PNGs")
        
        return {"message": "Generated MKV video in outputs folder"}
    except Exception as e:
        logger.error(f"Error generating MKV: {str(e)}")
        raise Exception(f"Error generating MKV: {str(e)}")

# Endpoint 3: Generate Map
@app.route("/generate_map", methods=["POST"])
def generate_map():
    if race_data is None or race_geojson is None:
        return jsonify({"error": "No geocoded race data found. Please upload and set locations first."}), 400
    
    # Parse query parameters
    format_type = request.args.get("format", "").lower()
    duration = request.args.get("duration", 10, type=float) if format_type in ["mkv", "svgs"] else None
    
    # Fallback to JSON body
    if not format_type:
        data = request.get_json(silent=True)
        if not data or "format" not in data:
            return jsonify({"error": "Missing 'format' parameter in query or JSON body"}), 400
        format_type = data["format"].lower()
        duration = float(data.get("duration", 10)) if format_type in ["mkv", "svgs"] else None
    
    if format_type not in ["svg", "mkv", "svgs"]:
        return jsonify({"error": "Format must be 'svg', 'mkv', or 'svgs'"}), 400
    
    try:
        if format_type == "svg":
            result = generate_svg_map(race_data)
        elif format_type == "mkv":
            result = generate_mkv_map(race_data, duration)
        else:  # svgs
            result = generate_svgs(race_data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

    
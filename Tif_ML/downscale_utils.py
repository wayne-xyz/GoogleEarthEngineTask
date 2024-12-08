import os
import rasterio
from rasterio.enums import Resampling
import numpy as np

def downscaling(input_tif, output_tif, matching_tif, bounds_tolerance=1e-4):
    """
    Downscale an input GeoTIFF to match the resolution and dimensions of another GeoTIFF, 
    preserving geospatial alignment, and save the result to an output GeoTIFF.
    
    Args:
        input_tif (str): Path to the input GeoTIFF file to be downscaled.
        output_tif (str): Path where the downscaled GeoTIFF will be saved.
        matching_tif (str): Path to the reference GeoTIFF file whose resolution and size will be matched.
        bounds_tolerance (float): Maximum allowed difference in bounds coordinates (in degrees).
    """
    # Open the reference (matching) GeoTIFF
    with rasterio.open(matching_tif) as match_src:
        # Extract target height, width, transform, and CRS from the reference
        target_height = match_src.height
        target_width = match_src.width
        target_transform = match_src.transform
        target_crs = match_src.crs
        target_bounds = match_src.bounds

    # Open the input GeoTIFF
    with rasterio.open(input_tif) as input_src:
        # Ensure the CRS match
        if input_src.crs != target_crs:
            raise ValueError("CRS mismatch between input_tif and matching_tif.")
        
        # Check bounds with tolerance
        input_bounds = input_src.bounds
        bounds_diff = [
            abs(input_bounds.left - target_bounds.left),
            abs(input_bounds.bottom - target_bounds.bottom),
            abs(input_bounds.right - target_bounds.right),
            abs(input_bounds.top - target_bounds.top)
        ]
        
        if max(bounds_diff) > bounds_tolerance:
            print("Input bounds:", input_bounds)
            print("Target bounds:", target_bounds)
            print("Maximum bounds difference:", max(bounds_diff))
            raise ValueError(f"Bounds mismatch exceeds tolerance of {bounds_tolerance} degrees.")
        
        # Perform the resampling to match dimensions
        data = input_src.read(
            out_shape=(input_src.count, target_height, target_width),
            resampling=Resampling.average
        )
        
        # Update the output profile to match the reference
        output_profile = input_src.profile.copy()
        output_profile.update({
            'height': target_height,
            'width': target_width,
            'transform': target_transform,
            'crs': target_crs
        })
        
        # Write the resampled data to the output GeoTIFF
        with rasterio.open(output_tif, 'w', **output_profile) as dst:
            dst.write(data)

def process_all_tifs(input_folder, output_folder, matching_folder, bounds_tolerance=1e-4):
    """
    Process all TIF files in the input folder.
    
    Args:
        input_folder (str): Path to input NICFI TIF folder
        output_folder (str): Path to output downscaled NICFI TIF folder
        matching_folder (str): Path to Sentinel TIF folder for matching schema
        bounds_tolerance (float): Maximum allowed difference in bounds coordinates
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the list of TIF files in the input folder
    input_tifs = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

    # Process each input TIF file
    for input_tif in input_tifs:
        input_path = os.path.join(input_folder, input_tif)
        output_path = os.path.join(output_folder, input_tif.replace('.tif', '_rs.tif'))

        # Get the file name of the matching TIF file
        match_tif = input_tif.split('-')[0] + "-" + input_tif.split('-')[1] + input_tif.split('-')[2] + '01-sentinel.tif'
        matching_path = os.path.join(matching_folder, match_tif)
        
        try:
            downscaling(input_path, output_path, matching_path, bounds_tolerance)
            print(f"Successfully processed: {input_tif}")
            print(f"Saved to: {output_path}")
            print(f"Matched with: {matching_path}")
        except Exception as e:
            print(f"Error processing {input_tif}: {str(e)}")

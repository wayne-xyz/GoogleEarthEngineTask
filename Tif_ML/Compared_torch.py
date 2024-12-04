import os
import torch
import pandas as pd
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def calculate_missing_percentage(data_tensor):
    """
    Calculate the percentage of missing (NaN) values in a given tensor.
    
    Args:
        data_tensor (torch.Tensor): The data tensor to check for missing values.

    Returns:
        float: The percentage of missing (NaN) values.
    """
    return (torch.isnan(data_tensor).sum() / torch.numel(data_tensor) * 100).item()

def load_and_process_image(file_path, device):
    """
    Load and process an image file using rasterio and convert to PyTorch tensor.
    
    Args:
        file_path (str): Path to the image file.
        device (torch.device): Device to load the tensor to.

    Returns:
        torch.Tensor: Processed image tensor on specified device.
    """
    with rasterio.open(file_path) as src:
        data = src.read()
        transform = src.transform
    return torch.tensor(data, dtype=torch.float32, device=device), transform

def get_coordinates(height, width, transform, device):
    """
    Generate coordinates for each pixel.
    
    Args:
        height (int): Image height.
        width (int): Image width.
        transform (Affine): Rasterio transform object.
        device (torch.device): Device to create tensors on.

    Returns:
        tuple: Tensors of latitudes and longitudes.
    """
    rows = torch.arange(height, device=device)
    cols = torch.arange(width, device=device)
    rows, cols = torch.meshgrid(rows, cols, indexing='ij')
    
    # Move to CPU for coordinate transformation
    rows_cpu = rows.cpu().numpy()
    cols_cpu = cols.cpu().numpy()
    
    lats = np.zeros((height, width))
    lons = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            lon, lat = transform * (cols_cpu[i, j], rows_cpu[i, j])
            lats[i, j] = lat
            lons[i, j] = lon
            
    return (torch.tensor(lats, device=device), 
            torch.tensor(lons, device=device))

def process_batch(batch_indices, nicfi_data, sentinel_data, target_data, 
                 lats, lons, height, width, row_date, row_dp_index):
    """
    Process a batch of pixels in parallel on GPU.
    
    Args:
        batch_indices (torch.Tensor): Indices of pixels to process.
        nicfi_data (torch.Tensor): NICFI image data.
        sentinel_data (list): List of Sentinel image data tensors.
        target_data (torch.Tensor): Target data.
        lats (torch.Tensor): Latitude values.
        lons (torch.Tensor): Longitude values.
        height (int): Image height.
        width (int): Image width.
        row_date (str): Date string.
        row_dp_index (str): DP index string.

    Returns:
        list: List of dictionaries containing processed pixel data.
    """
    rows = batch_indices // width
    cols = batch_indices % width
    
    batch_size = len(batch_indices)
    rows_list = []
    
    # Get all values for the batch
    nicfi_values = nicfi_data[:, rows, cols]  # Shape: (4, batch_size)
    sentinel_values = [s[:, rows, cols] for s in sentinel_data]  # Shape: (3, 26, batch_size)
    target_values = target_data[rows, cols]  # Shape: (batch_size,)
    lat_values = lats[rows, cols]  # Shape: (batch_size,)
    lon_values = lons[rows, cols]  # Shape: (batch_size,)
    
    # Move tensors to CPU for numpy operations
    nicfi_values = nicfi_values.cpu().numpy()
    sentinel_values = [s.cpu().numpy() for s in sentinel_values]
    target_values = target_values.cpu().numpy()
    lat_values = lat_values.cpu().numpy()
    lon_values = lon_values.cpu().numpy()
    
    for i in range(batch_size):
        pixel_idx = batch_indices[i].item()
        
        # Handle NICFI bands
        nicfi_dict = {
            f'nicfi_band{b+1}': (float(nicfi_values[b, i]) 
                                if not np.isnan(nicfi_values[b, i]) else None)
            for b in range(4)
        }
        
        # Handle Sentinel bands
        sentinel_dict = {
            f's{s_idx+1}_band{b+1}': (float(sentinel_values[s_idx][b, i])
                                     if not np.isnan(sentinel_values[s_idx][b, i]) else None)
            for s_idx in range(3) for b in range(26)
        }
        
        row_data = {
            'dp_index': row_dp_index,
            'pixel_index': pixel_idx + 1,
            'date': f"{row_date[0:4]}-{row_date[4:6]}",
            'latitude': float(lat_values[i]),
            'longitude': float(lon_values[i]),
            **nicfi_dict,
            **sentinel_dict,
            'target': float(target_values[i]) if not np.isnan(target_values[i]) else None
        }
        rows_list.append(row_data)
    
    return rows_list

def save_to_parquet(df, parquet_save_path):
    """
    Save DataFrame to Parquet file with proper error handling and append functionality.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        parquet_save_path (str): Path to save the Parquet file
    """
    try:
        # If file doesn't exist or is empty, write directly
        if not os.path.exists(parquet_save_path) or os.path.getsize(parquet_save_path) == 0:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(parquet_save_path), exist_ok=True)
            # Write new DataFrame
            df.to_parquet(parquet_save_path, engine='pyarrow', index=False)
            print(f"Created new Parquet file at {parquet_save_path}")
        else:
            try:
                # Try to read existing file
                existing_df = pd.read_parquet(parquet_save_path, engine='pyarrow')
                # Append new data
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                # Save combined data
                combined_df.to_parquet(parquet_save_path, engine='pyarrow', index=False)
                print(f"Appended data to existing Parquet file at {parquet_save_path}")
            except Exception as e:
                print(f"Error reading existing Parquet file: {e}")
                print("Backing up old file and creating new one...")
                # Backup old file
                backup_path = parquet_save_path + '.backup'
                if os.path.exists(parquet_save_path):
                    os.rename(parquet_save_path, backup_path)
                # Write new file
                df.to_parquet(parquet_save_path, engine='pyarrow', index=False)
                print(f"Created new Parquet file at {parquet_save_path}")
    except Exception as e:
        print(f"Error saving to Parquet: {e}")
        # Save to CSV as fallback
        csv_path = parquet_save_path.replace('.parquet', '.csv')
        print(f"Attempting to save as CSV to: {csv_path}")
        df.to_csv(csv_path, index=False)

def tabularize_bands_with_torch_to_parquet(nicfi_folder_path, upscaled_sentinel_folder_path, 
                                         target_folder_path, parquet_save_path, 
                                         batch_size=1000):
    """
    Tabularize the bands of the NICFI and Upscaled Sentinel images using PyTorch for GPU-accelerated
    numerical operations, and save the data in Parquet format.
    
    Args:
        nicfi_folder_path (str): Path to the NICFI folder.
        upscaled_sentinel_folder_path (str): Path to the Upscaled Sentinel folder.
        target_folder_path (str): Path to the target folder.
        parquet_save_path (str): Path to save the Parquet file.
        batch_size (int): Number of pixels to process in parallel.
    """
    # Column design
    parquet_columns = [
        'dp_index', 'pixel_index', 'date', 'latitude', 'longitude',
        *[f'nicfi_band{i+1}' for i in range(4)],
        *[f's{i+1}_band{j+1}' for i in range(3) for j in range(26)],
        'target'
    ]
    
    for nicfi_file in tqdm(os.listdir(nicfi_folder_path), desc="Processing files"):
        nicfi_path = os.path.join(nicfi_folder_path, nicfi_file)
        nicfi_file_name = Path(nicfi_file).name
        row_date = nicfi_file_name.split('-')[1] + nicfi_file_name.split('-')[2]
        row_dp_index = nicfi_file_name.split('-')[0]

        target_file = f"{row_dp_index}-{row_date[0:4]}-{row_date[4:6]}-target.tif"
        target_path = os.path.join(target_folder_path, target_file)

        if not os.path.exists(target_path):
            print(f"Target file not found: {target_file}")
            continue

        upscaled_sentinel_files = [
            f for f in os.listdir(upscaled_sentinel_folder_path)
            if row_dp_index == f.split('-')[0] and row_date in f
        ]
        upscaled_sentinel_files.sort()

        if len(upscaled_sentinel_files) != 3:
            print(f"Expected 3 upscaled sentinel files for {row_date}, but found {len(upscaled_sentinel_files)}")
            continue

        # Load all data to GPU
        nicfi_data, transform = load_and_process_image(nicfi_path, device)
        height, width = nicfi_data.shape[1:]
        
        target_data, _ = load_and_process_image(target_path, device)
        target_data = target_data[0]  # Take first band only
        
        sentinel_data = []
        for sentinel_file in upscaled_sentinel_files:
            data, _ = load_and_process_image(
                os.path.join(upscaled_sentinel_folder_path, sentinel_file), 
                device
            )
            sentinel_data.append(data)

        # Generate coordinates
        lats, lons = get_coordinates(height, width, transform, device)
        
        # Process in batches
        all_pixels = torch.arange(height * width, device=device)
        all_rows = []
        
        for i in range(0, len(all_pixels), batch_size):
            batch_indices = all_pixels[i:i + batch_size]
            batch_rows = process_batch(
                batch_indices, nicfi_data, sentinel_data, target_data,
                lats, lons, height, width, row_date, row_dp_index
            )
            all_rows.extend(batch_rows)

        df = pd.DataFrame(all_rows, columns=parquet_columns)

        # Calculate missing data percentage
        nicfi_missing = calculate_missing_percentage(nicfi_data)
        sentinel_missing = sum(calculate_missing_percentage(s) for s in sentinel_data) / 3
        target_missing = calculate_missing_percentage(target_data)

        print(f"\nProcessed {nicfi_file} with {len(all_rows)} pixels")
        print(f"NICFI bands missing data: {nicfi_missing:.2f}%")
        print(f"Sentinel bands missing data: {sentinel_missing:.2f}%")
        print(f"Target data missing: {target_missing:.2f}%")

        # Save to Parquet
        save_to_parquet(df, parquet_save_path)

if __name__ == "__main__":
    # Example usage
    nicfi_folder = "path/to/nicfi"
    sentinel_folder = "path/to/sentinel"
    target_folder = "path/to/target"
    output_file = "output.parquet"
    
    tabularize_bands_with_torch_to_parquet(
        nicfi_folder, sentinel_folder, target_folder, output_file
    )

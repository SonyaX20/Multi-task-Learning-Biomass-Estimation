import os
import requests
from tqdm import tqdm

def download_file(url, destination):
    """
    Download a file from a URL to a specified destination with progress bar
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(destination):
        print(f"File already exists at {destination}")
        return
    
    # Stream the download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Get file size if available
    total_size = int(response.headers.get('content-length', 0))
    
    # Create progress bar
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
    
    # Write the file
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))
    
    progress_bar.close()
    print(f"Download completed: {destination}")

# URL of the biomass data
url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/BIOMASS/SUSBIOM/LATEST/Biomass/AGB_2020_EU27.tif"

# Destination path
destination = "data/biomass/AGB_2020_EU27.tif"

# Download the file
download_file(url, destination)

import xarray as xr
import dask.array as da
import json
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import numpy as np

def setup_dask_cluster():
    # Create a local Dask cluster
    cluster = LocalCluster(
        n_workers=8,              # Use physical cores
        threads_per_worker=2,     # Utilize hyper-threading
        memory_limit='40GB'        # Memory limit per worker
    )
    client = Client(cluster)
    return client

def compute_stats():
    # Set up the Dask cluster
    client = setup_dask_cluster()
    
    # Load the dataset
    ds = xr.open_mfdataset('data/daily_processed/*.nc', chunks={'time': 100000000}, combine='by_coords')

    # Initialize dictionaries for mean and standard deviation values
    mean_values = {}
    std_values = {}

    # Setup the progress bar and compute statistics
    with ProgressBar():
        for var in ds.data_vars:
            if ds[var].dtype in [np.float32, np.float64]:
                # Compute mean and standard deviation, showing progress
                mean_values[var] = ds[var].mean(dim='time').compute()
                std_values[var] = ds[var].std(dim='time').compute()

    # Convert results to float for JSON serialization
    normalization_values = {
        'mean': {k: float(v) for k, v in mean_values.items()},
        'std': {k: float(v) for k, v in std_values.items()}
    }

    # Save the normalization values to a JSON file
    with open('normalization_values.json', 'w') as f:
        json.dump(normalization_values, f)

    print("Normalization values saved to normalization_values.json")
    
    # Close the client and shutdown the cluster when done
    client.close()

if __name__ == "__main__":
    compute_stats()
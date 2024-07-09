import os
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

def normalize_data(data, max_values):
    normalized_data = {}
    for var, value in data.items():
        normalized_data[var] = value / 2000
    return normalized_data

class NCDataset(Dataset):
    def __init__(self, folder_path, start_hour, end_hour):
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
        self.start_hour = start_hour
        self.end_hour = end_hour

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = os.path.join(self.folder_path, self.file_list[index])
        ds = xr.open_dataset(file_path)
        ds = xr.decode_cf(ds)
        # Get the length of the time dimension
        time_length = 1440

        # Calculate the indices for splitting the data based on start and end hours
        start_index = int(time_length * self.start_hour / 24)
        end_index = int(time_length * self.end_hour / 24)

        # Extract the desired variables from the dataset within the time range
        response = {
            'GHI_HR': ds['GHI'].isel(time=slice(start_index, end_index)).values,
            #'BNI_HR': ds['BNI'].isel(time=slice(start_index, end_index)).values,
            # Add more variables as needed
        }
            # Resample and interpolate GHI_HR
        ghi_hr_15min = ds['GHI'].values.reshape(-1, 15).mean(axis=1)
        ghi_hr_1min = np.interp(np.linspace(0, len(ghi_hr_15min) - 1, time_length), np.arange(len(ghi_hr_15min)), ghi_hr_15min)
        
        cont_condition = {
            'GHI_HR': ghi_hr_1min[start_index:end_index],
            'GHI_CS': ds['ghi_clear'].isel(time=slice(start_index, end_index)).values,
            # Add more variables as needed
        }
        
        cat_condition = {}

        # Normalize the data using the normalization function
        normalized_response = normalize_data(response, 2000)
        
        if cont_condition:
            normalized_cont_condition = normalize_data(cont_condition, 2000)
            cont_condition = {var: torch.tensor(value, dtype=torch.float32) for var, value in normalized_cont_condition.items()}
        
        if cat_condition:
            normalized_cat_condition = normalize_data(cat_condition, 2000)
            cat_condition = {var: torch.tensor(value, dtype=torch.float32) for var, value in normalized_cat_condition.items()}

        # Convert the normalized data to PyTorch tensors
        tensor_response = {var: torch.tensor(value, dtype=torch.float32) for var, value in normalized_response.items()}

        return {'response': tensor_response, 'cont_condition': cont_condition, 'cat_condition': cat_condition}

def custom_collate_fn(batch, start_hour, end_hour):
    # Calculate the expected sequence length based on start and end hours
    expected_length = int(1440 * (end_hour - start_hour) / 24)

    # Filter out sequences that don't match the expected length
    filtered_batch = [sample for sample in batch if sample['response']['GHI_HR'].shape[0] == expected_length]

    # Collate the filtered batch
    collated_batch = {
        'response': {var: torch.stack([sample['response'][var] for sample in filtered_batch]) for var in filtered_batch[0]['response']},
        'cont_condition': {var: torch.stack([sample['cont_condition'][var] for sample in filtered_batch]) for var in filtered_batch[0]['cont_condition']} if filtered_batch[0]['cont_condition'] else None,
        'cat_condition': {var: torch.stack([sample['cat_condition'][var] for sample in filtered_batch]) for var in filtered_batch[0]['cat_condition']} if filtered_batch[0]['cat_condition'] else None
    }
    return collated_batch

def get_data_loaders(folder_path, start_hour, end_hour, batch_size, train_ratio=0.8):
    dataset = NCDataset(folder_path, start_hour, end_hour)
    
    # Split the dataset into train and validation sets
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch, start_hour, end_hour))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: custom_collate_fn(batch, start_hour, end_hour))
    
    return train_loader, val_loader
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


def create_or_append_to_xr_dataset(
    data_dict: dict[str, np.ndarray], cache_file: str
) -> xr.Dataset:
    """
    Constructs or appends to an xarray.Dataset from a dictionary of numpy arrays,
    ensuring dimension uniqueness by renaming, with 'batch_dim' labeled for the first dimension.
    This function writes to a Zarr store on-the-fly.

    Parameters:
        data_dict: A dictionary where the keys are strings representing variable names
                   and the values are numpy arrays representing the data.
        cache_file: The path where the Zarr store will be saved (dir/filename.zarr).

    Returns:
        An xarray.Dataset constructed or appended to with uniquely renamed dimensions,
        with the first dimension labeled as 'batch_dim' and stored on disk.
    """

    data_arrays = [
        xr.DataArray(data=value, name=key).rename(
            {
                da.dims[0]: "batch_dim",
                **{dim: f"{key}_dim{i}" for i, dim in enumerate(da.dims) if i > 0},
            }
        )
        for key, value in data_dict.items()
        for da in [xr.DataArray(data=value)]
    ]

    # Append dataset to Zarr store
    dataset = xr.Dataset({da.name: da for da in data_arrays})
    dataset.to_zarr(cache_file, mode="a", consolidated=False)
    return dataset


class CacheDataset(Dataset):
    """Dataset for loading the cached latents and encodings."""

    def __init__(self, cache_file: str, dtype: torch.dtype):
        """Loads the cached encodings in read mode for faster initialization.
        Data is converted into torch.Tensors of given `dtype`.

        Args:
            cache_file (str): The path to the cached zarr file containing e.g latents and text encodings.
            dtype (torch.dtype): The datatype to cast the data into.
        """
        self.cached_dataset = xr.open_zarr(cache_file, consolidated=False)
        self.keys = self.cached_dataset.data_vars.keys()
        self.dtype = dtype

        # Check same length for all arrays using 'batch_dim'
        length_set = set(var.sizes["batch_dim"] for var in self.cached_dataset.values())
        assert len(length_set) == 1, f"Arrays have different lengths: {length_set}"

    def __len__(self):
        return self.cached_dataset.sizes["batch_dim"]

    def __getitem__(self, idx):
        output_data = {}
        # Use .isel() for index-based (lazy) selection
        batch_slice = self.cached_dataset.isel(batch_dim=idx).compute(
            scheduler="synchronous"
        )
        for key in self.keys:
            array = batch_slice[key].values.astype(np.float32)
            output_data[key] = torch.from_numpy(array).to(self.dtype)

        return output_data

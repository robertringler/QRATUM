"""HDF5 I/O utilities for QuASIM state management."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Use h5py if available, otherwise provide graceful fallback
try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


def write_snapshot(path: str, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
    """Write snapshot to HDF5 file with metadata and arrays.

    Parameters
    ----------
    path : str
        Output HDF5 file path
    meta : Dict[str, Any]
        Metadata dictionary (stored as attributes)
    arrays : Dict[str, np.ndarray]
        Data arrays to store as datasets

    Examples
    --------
    >>> import numpy as np
    >>> write_snapshot(
    ...     "test.hdf5",
    ...     {"version": "1.0", "seed": 42},
    ...     {"data": np.array([1, 2, 3])}
    ... )
    """

    if not HDF5_AVAILABLE:
        # Fallback: write metadata as JSON
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        meta_path = Path(path).with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))

        # Write arrays as .npy files
        for key, arr in arrays.items():
            npy_path = Path(path).parent / f"{key}.npy"
            np.save(npy_path, arr)
        return

    # Create parent directory
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Write metadata group
        fm = f.create_group("meta")
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                fm.attrs[k] = v
            else:
                # Store complex objects as JSON strings
                fm.attrs[k] = json.dumps(v)

        # Write arrays as datasets with compression
        for key, arr in arrays.items():
            if arr.size > 0:
                f.create_dataset(key, data=arr, compression="gzip", shuffle=True, fletcher32=True)
            else:
                # Handle empty arrays
                f.create_dataset(key, data=arr)


def read_snapshot(path: str) -> Dict[str, Any]:
    """Read snapshot from HDF5 file.

    Parameters
    ----------
    path : str
        Input HDF5 file path

    Returns
    -------
    Dict[str, Any]
        Dictionary containing 'meta' and dataset keys

    Examples
    --------
    >>> data = read_snapshot("test.hdf5")
    >>> print(data['meta']['seed'])
    42
    """

    if not HDF5_AVAILABLE:
        # Fallback: read from JSON and .npy files
        meta_path = Path(path).with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        result = {"meta": meta}

        # Try to load .npy files
        for npy_path in Path(path).parent.glob("*.npy"):
            key = npy_path.stem
            result[key] = np.load(npy_path)

        return result

    result = {}

    with h5py.File(path, "r") as f:
        # Read metadata
        if "meta" in f:
            meta = {}
            for k, v in f["meta"].attrs.items():
                try:
                    # Try to parse as JSON
                    meta[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    # Store as-is if not JSON
                    meta[k] = v
            result["meta"] = meta

        # Read all datasets
        for key in f:
            if key != "meta":
                result[key] = f[key][:]

    return result

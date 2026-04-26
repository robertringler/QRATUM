"""I/O handlers for simulation artifacts."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd


class ArtifactManager:
    """Manages simulation artifacts and outputs."""
    
    def __init__(self, base_dir: str = "runs"):
        """
        Initialize artifact manager.
        
        Args:
            base_dir: Base directory for runs
        """
        self.base_dir = Path(base_dir)
        self.run_dir = None
        
    def create_run_directory(self) -> Path:
        """
        Create a timestamped run directory.
        
        Returns:
            Path to run directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir
    
    def save_json(self, data: Dict[str, Any], filename: str):
        """Save JSON data to run directory."""
        if self.run_dir is None:
            raise ValueError("Run directory not created")
        
        filepath = self.run_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_dataframe(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to parquet format."""
        if self.run_dir is None:
            raise ValueError("Run directory not created")
        
        filepath = self.run_dir / filename
        df.to_parquet(filepath, index=False)
    
    def save_figure(self, fig: plt.Figure, filename: str):
        """Save matplotlib figure."""
        if self.run_dir is None:
            raise ValueError("Run directory not created")
        
        filepath = self.run_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def get_run_path(self) -> Path:
        """Get the current run directory path."""
        if self.run_dir is None:
            raise ValueError("Run directory not created")
        return self.run_dir

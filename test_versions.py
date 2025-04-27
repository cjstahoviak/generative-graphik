# Save this as test_imports.py
import torch
print(f"PyTorch version: {torch.__version__}")

import torch_sparse
print("Successfully imported torch_sparse")

import torch_geometric
print(f"PyG version: {torch_geometric.__version__}")

from torch_geometric.data import Data
print("Successfully imported Data")

print("All imports successful!")

# Python version
import sys
print(f"Python version: {sys.version}")
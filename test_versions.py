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

# Test if cuda is available and if pyTorch is built with CUDA
if torch.cuda.is_available():
    print("CUDA is available")
    print("PyTorch is built with CUDA")
else:
    print("CUDA is not available")
    print("PyTorch is not built with CUDA")

# Test if graphik is installed
import importlib, graphik, pathlib, sys
print(f"Found GraphIK")
print(f"Using graphik from: {graphik.__file__}")

print(list(graphik.__path__))          # each entry is one directory on disk
# Same information, but via the import machinery
spec = importlib.util.find_spec("graphik")
print(spec.origin)                     # will say 'namespace'
print(list(spec.submodule_search_locations))
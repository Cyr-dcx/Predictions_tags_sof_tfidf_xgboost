import os
from pathlib import Path 

TEST_DIR = str(Path(utils_package.__file__))
TEST1_DIR = str(Path(utils_package.__file__).parent)
ROOT_DIR = str(Path(utils_package.__file__).parent.parent)

print(TEST_DIR)
print(TEST1_DIR)
print(ROOT_DIR)
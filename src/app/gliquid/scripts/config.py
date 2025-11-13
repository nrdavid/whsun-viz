from pathlib import Path
import os
import zipfile
_DIR_STRUCT_OPTS = ['flat', 'nested']

project_root = None
data_dir = None
dir_structure = None
fusion_enthalpies_file = None
fusion_temps_file = None
vaporization_temps_file = None

def set_project_root(path: Path):
    global project_root
    project_root = path

def set_data_dir(path: Path):
    global data_dir
    global fusion_enthalpies_file
    global fusion_temps_file
    global vaporization_temps_file

    data_dir = path
    fusion_enthalpies_file =  Path(data_dir / "fusion_enthalpies.json")
    fusion_temps_file = Path(data_dir / "fusion_temperatures.json")
    vaporization_temps_file = Path(data_dir / "vaporization_temperatures.json")

def set_dir_structure(structure: str):
    global dir_structure
    if structure not in _DIR_STRUCT_OPTS:
        raise ValueError(f"dir_structure must be one of {_DIR_STRUCT_OPTS}")
    dir_structure = structure

set_project_root(Path.cwd())
set_data_dir(Path(project_root / "gliquid/binary_cache"))
set_dir_structure(_DIR_STRUCT_OPTS[0])

for zip_path in ["gliquid/matrix_assets.zip", "gliquid/binary_cache.zip"]:
    target_dir = zip_path.replace('.zip', '')
    if os.path.exists(zip_path) and not os.path.exists(target_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(target_dir))
    assert os.path.exists(target_dir), f"Required directory {target_dir} does not exist"
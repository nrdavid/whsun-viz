from pathlib import Path
_DIR_STRUCT_OPTS = ['flat', 'nested']

project_root = None
data_dir = None
dir_structure = None

def set_project_root(path: Path):
    global project_root
    project_root = path

def set_data_dir(path: Path):
    global data_dir
    data_dir = path

def set_dir_structure(structure: str):
    global dir_structure
    if structure not in _DIR_STRUCT_OPTS:
        raise ValueError(f"dir_structure must be one of {_DIR_STRUCT_OPTS}")
    dir_structure = structure

set_project_root(Path.cwd())
set_data_dir(Path(project_root / "gliquid_ternary_interpolation/matrix_data_jsons"))
set_dir_structure(_DIR_STRUCT_OPTS[1])

fusion_enthalpies_file = data_dir / "fusion_enthalpies.json"
fusion_temps_file = data_dir / "fusion_temperatures.json"
vaporization_temps_file = data_dir / "vaporization_temperatures.json"

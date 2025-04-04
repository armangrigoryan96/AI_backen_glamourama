import subprocess
import sys
import os
from tqdm import tqdm
def run_command(command):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.stdout
    except Exception as e:
        print(f"Error: {e}")
def commit_and_push(file, message, branch='main'):
    """Commit specific files and push to GitHub."""
    
    res = run_command(f"git add {file}")
    if res.startswith("Error"):
        return "Error"
    print(f"Committing changes with message: {message}")
    res = run_command(f'git commit -m "{message}"')
    if res.startswith("Error"):
        return  "Error"
    
    print(f"Pushing changes to branch: {branch}")
    res =run_command(f"git push origin {branch}")
    if res.startswith("Error"):
        return "Error"
    return "Success"

files=[
    '=1.4.1', 
    'config_files', 
    'images', 
    'main.py', 
    'measure_second.py', 
    'run_demo.sh', 
    'transfer_model', 
    'visualize.py',
    '=0.15', 
    '=4.1.1', 
    'configs', 
    'ip_adapter', 
    'main_origin.py', 
    'measurement_definitions.py', 
    'start.py', 
    'tryon_dress',
    '=0.26.1', 
    'Anaconda3-2022.05-Linux-x86_64.sh', 
    'data', 
    'joint_definitions.py', 
    'measure.py', 
    'measurement_definitions_origin.py', 
    'start.py.save', 
    'tryonvenv',
    '=0.69', 
    'base', 
    'engine', 
    'landmark_definitions.py', 
    'measure.txt', 
    'output', 
    'start_origin.py', 
    'utils.py',
    '=1.23.0', 
    'ckpt', 
    'evaluate.py', 
    'load_pkl.py', 
    'measure_origin.py', 
    'pose_parsing', 
    'transfer_data', 
    'utils_origin.py'
]

message = "initial commit"
for i in tqdm(range(len(files))):
    file = files[i]
    print(f"{i}: ", end = " ")
    if not os.path.exists(file):
        print(f"File {file} does not exist.")
    else:
        print(f"File {file} exists.")

    res = commit_and_push(file, message)
    print(res)
commit_and_push(files, message)

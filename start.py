import subprocess
import time
import argparse
import sys
import json
import os

parser = argparse.ArgumentParser(description='Measurements')
parser.add_argument('-ht', '--height', type=str, default='170.5', help='Custome height of body')

args=parser.parse_args()

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files..")

# clear previous working
#input_model_path = '3d_body/inputs'
#output_smplx_path = '3d_body/results'
#output_smpl_path = 'output'
#delete_files_in_directory(input_model_path)
#delete_files_in_directory(output_smplx_path)
#delete_files_in_directory(output_smpl_path)

# Activate the Conda environment
activate_cmd = f'eval "$(conda shell.bash hook)" && conda activate 3d_body && '

# Execute the script within the environment
subprocess.run(activate_cmd + 'python ./engine/demos/demo_fit_body.py', shell=True, check=True)

# Run the child script and capture its output
try:
    # Activate the Conda environment
    #activate_cmd = f'conda activate trans && '
    activate_cmd = f'eval "$(conda shell.bash hook)" && conda activate measurement && '
    result = subprocess.run(activate_cmd + 'python main.py ' + '--height ' + args.height, shell=True, capture_output=True, text=True)
    # Check if the command ran successfully
    if result.returncode == 0:
        measurements = result.stdout
        print(measurements)
    else:
        print('Error running the child script:')
        print(result.stderr)
except Exception as e:
    print('Error:', e)   



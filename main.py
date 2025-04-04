#import sys
#sys.path.append(r"F:/Task/222/6_3D_Human_body_reconstruction/reference/measurements/SMPL-Anthropometry")

# main.py
import subprocess
import measure
import argparse
import sys
import json
sys.setrecursionlimit(20000)

parser = argparse.ArgumentParser(description='Measurements')
parser.add_argument('-ht', '--height', type=str, default='170.5', help='Custome height of body')

args=parser.parse_args()

#from .smplx.transfer_model import __main__
#from transfer_model import app

# Run the first file
transfer_process = subprocess.Popen(['python', '-m','transfer_model'])
transfer_process.wait()
# Run the second file

result = subprocess.run(['python', 'measure.py', '--height', args.height], capture_output=True, text=True)
#measure_process.wait()
'''
if result.returncode == 0:
    # Deserialize the JSON string received from the child script
    measurements = json.loads(result.stdout)
    
    # Print the measurements dictionary
    print("Measurements received from measurement script:")
    print(measurements)
else:
    # Print an error message if the child script failed
    print("Error running measurement script. Return code:", result.returncode)
    print("Error message:", result.stderr)
'''
print(result.stdout)


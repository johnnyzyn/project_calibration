import argparse
import subprocess
import os
from robustness_dataset import RobustnessDataset
import numpy as np

parser = argparse.ArgumentParser(description="Loop through archi_num and run main.py")
parser.add_argument("--reset", action="store_true", help="Remove the specified CSV file before starting the loop")
parser.add_argument("--csv_file", type=str, default="results.csv", help="CSV file name to store or update results")
parser.add_argument("--image_dataset", type=str, default="cifar10", help="Image dataset to use: CIFAR-10, CIFAR-100, or ImageNet16-120")

args = parser.parse_args()

data = RobustnessDataset(path="/home/younan/project_calibration")

if args.reset:
    if os.path.exists(args.csv_file):
        os.remove(args.csv_file)

# Open the error log file in append mode

# Find the index of the last processed ID in the list
unique_ids = data.non_isomorph_ids
last_processed_index = unique_ids.index('0')

# Continue processing from the next ID in the list


with open("error_log.txt", "a") as error_log:
    for i in range(last_processed_index + 1, len(unique_ids)):
    # for i in range(len(unique_ids)):
        archi_num = unique_ids[i]
        try:
            result = subprocess.run(["python", "main.py", "--archi_num", str(archi_num), "--csv_file", args.csv_file, "--image_dataset", args.image_dataset], stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                error_msg = f"Error occurred for archi_num: {archi_num}. Error: {result.stderr}\n"
                print(error_msg)
                # Write the error message to the error log file
                error_log.write(error_msg)
        except Exception as e:
            error_msg = f"Subprocess error for archi_num: {archi_num}. Error: {e}\n"
            print(error_msg)
            # Write the subprocess error message to the error log file
            error_log.write(error_msg)
            continue
# python run_main.py --csv_file cifar10_results.csv --image_dataset cifar10
# python run_main.py --csv_file cifar100_results.csv --image_dataset cifar100
# python run_main.py --csv_file imagenet_results.csv --image_dataset ImageNet16-120


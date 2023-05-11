import argparse
import subprocess
import os
from robustness_dataset import RobustnessDataset
import numpy as np

parser = argparse.ArgumentParser(description="Loop through archi_num and run main.py")
parser.add_argument("--reset", action="store_true", help="Remove the specified CSV file before starting the loop")
parser.add_argument("--csv_file", type=str, default="results.csv", help="CSV file name to store or update results")
parser.add_argument("--image_dataset", type=str, default="cifar10", help="Image dataset to use: CIFAR-10, CIFAR-100, or ImageNet16-120")
parser.add_argument("--api_type", type=str, default="tss", help="NATS-Bench dataset type (tss or sss)")
parser.add_argument("--post_temp", type=str, default='False', help="if using temp scale")
parser.add_argument("--device", type=str, default='cuda:0', help="device")
parser.add_argument("--archi_start", type=int, default=0, help="archi number to start")
parser.add_argument("--archi_end", type=int, default=10000, help="archi number to end")

args = parser.parse_args()

data = RobustnessDataset(path="/home/younan/project_calibration")

if args.reset:
    if os.path.exists(args.csv_file):
        os.remove(args.csv_file)

# Open the error log file in append mode

# Find the index of the last processed ID in the list
# unique_ids = data.non_isomorph_ids
# last_processed_index = unique_ids.index('0')

# Continue processing from the next ID in the list


with open("error_log.txt", "a") as error_log:
    # for i in range(last_processed_index + 1, len(unique_ids)):
    for archi_num in range(args.archi_start, args.archi_end):
        try:
            result = subprocess.run(["python", "sss.py"
                                     , "--archi_num", str(archi_num)
                                     , "--csv_file", args.csv_file
                                     , "--image_dataset", args.image_dataset
                                     ,"--post_temp",args.post_temp
                                     ,"--api_type",args.api_type
                                     ,"--device",args.device], stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                error_msg = f"Error occurred for archi_num: {archi_num}, image_dataset: {args.image_dataset}, api_type: {args.api_type}, post_temp: {args.post_temp}. Error: {result.stderr}\n"
                print(error_msg)
                # Write the error message to the error log file
                error_log.write(error_msg)
        except Exception as e:
            error_msg = f"Subprocess error for archi_num: {archi_num}, image_dataset: {args.image_dataset}, api_type: {args.api_type}, post_temp: {args.post_temp}. Error: {e}\n"
            print(error_msg)
            # Write the subprocess error message to the error log file
            error_log.write(error_msg)
            continue


# python train_sss.py --csv_file sss/cifar10_sss_p1.csv --image_dataset cifar10 --api_type sss --device cuda:2 --archi_start 0 --archi_end 10000
# python train_sss.py --csv_file sss/cifar10_sss_p2.csv --image_dataset cifar10 --api_type sss --device cuda:2 --archi_start 10000 --archi_end 20000
# python train_sss.py --csv_file sss/cifar10_sss_p3.csv --image_dataset cifar10 --api_type sss --device cuda:2 --archi_start 20000 --archi_end 32768

# python train_sss.py --csv_file sss/cifar100_sss_p1.csv --image_dataset cifar100 --api_type sss --device cuda:3 --archi_start 0 --archi_end 5000
# python train_sss.py --csv_file sss/cifar100_sss_p2.csv --image_dataset cifar100 --api_type sss --device cuda:3 --archi_start 5000 --archi_end 10000
# python train_sss.py --csv_file sss/cifar100_sss_p3.csv --image_dataset cifar100 --api_type sss --device cuda:3 --archi_start 10000 --archi_end 15000
# python train_sss.py --csv_file sss/cifar100_sss_p4.csv --image_dataset cifar100 --api_type sss --device cuda:3 --archi_start 15000 --archi_end 20000
# python train_sss.py --csv_file sss/cifar100_sss_p5.csv --image_dataset cifar100 --api_type sss --device cuda:3 --archi_start 20000 --archi_end 25000
# python train_sss.py --csv_file sss/cifar100_sss_p6.csv --image_dataset cifar100 --api_type sss --device cuda:3 --archi_start 25000 --archi_end 32768

# python train_sss.py --csv_file sss/imagenet_sss_p1.csv --image_dataset ImageNet16-120 --api_type sss --device cuda:3 --archi_start 0 --archi_end 5000
# python train_sss.py --csv_file sss/imagenet_sss_p2.csv --image_dataset ImageNet16-120 --api_type sss --device cuda:3 --archi_start 5000 --archi_end 10000
# python train_sss.py --csv_file sss/imagenet_sss_p3.csv --image_dataset ImageNet16-120 --api_type sss --device cuda:3 --archi_start 10000 --archi_end 15000
# python train_sss.py --csv_file sss/imagenet_sss_p4.csv --image_dataset ImageNet16-120 --api_type sss --device cuda:3 --archi_start 15000 --archi_end 20000
# python train_sss.py --csv_file sss/imagenet_sss_p5.csv --image_dataset ImageNet16-120 --api_type sss --device cuda:3 --archi_start 20000 --archi_end 25000
# python train_sss.py --csv_file sss/imagenet_sss_p6.csv --image_dataset ImageNet16-120 --api_type sss --device cuda:3 --archi_start 25000 --archi_end 32768

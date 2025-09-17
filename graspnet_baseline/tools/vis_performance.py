import numpy as np
import os
import json
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuration ---
split = 'test_similar_mixed_mini'
# split = 'test_similar_mixed'
# split = 'test_similar'

num_pairs_to_process = 1000  # Set to None to process all pairs
dump_folder = '/home/seung/Workspaces/grasp/TestAdaGrasp/graspnet_baseline/logs'
root = '/home/seung/Workspaces/Datasets/GraspNet-1Billion'
# Methods to compare (exactly two)
# methods = ['graspnet1b/tta-grasp/exp_tmp/realsense_similar_mini_ens', 'graspnet1b/tta-grasp/exp_tmp/realsense_similar_mini_contrast']
methods = ['graspnet1b/notta/realsense_similar', 'graspnet1b/bn-adapt/realsense_similar_mixed_mini']
# methods = ['graspnet1b/tta-grasp/exp_bs1/realsense_similar_mixed', 'graspnet1b/tta-grasp/exp_bs1/realsense_similar_mixed_noconst']
methods = ['graspnet1b/notta/realsense_similar', 'graspnet1b/tta-grasp/realsense_similar_mixed_mini_q0.3_u0.05']
methods = ['graspnet1b/notta/realsense_similar', 'graspnet1b/tta-grasp/realsense_similar_mixed_mini_q0.3_u0.05']

method_names = ['notta', 'tta-grasp']  # Simplified names for display
sigma = 5  # Sigma for smoothing


# --- Data Loading ---
print(f"Loading data for split: {split}")
try:
    with open(os.path.join(root, 'splits', f'{split}.json'), 'r') as f:
        scene_id_img_id_pairs_all = json.load(f)
except FileNotFoundError:
    print(f"Error: Split file not found at {os.path.join(root, 'splits', f'{split}.json')}")
    exit()

# Select the pairs to process
if num_pairs_to_process is not None:
    scene_id_img_id_pairs = scene_id_img_id_pairs_all[:num_pairs_to_process]
    print(f"Processing the first {len(scene_id_img_id_pairs)} pairs...")
else:
    scene_id_img_id_pairs = scene_id_img_id_pairs_all
    print(f"Processing all {len(scene_id_img_id_pairs)} pairs...")

aps = {method: [] for method in methods}
# Iterate through scene/image pairs and load accuracy data
for scene_id, img_id in tqdm(scene_id_img_id_pairs, desc="Loading Accuracy Data"):
    for method in methods:
        acc_list_path = os.path.join(dump_folder, method, f'scene_{scene_id:04d}', 'realsense', f'{img_id:04d}_acc.npy')
        acc = np.load(acc_list_path)
        acc_mean = np.mean(acc, axis=0)
        ap_04 = acc_mean[1]*100
        ap_08 = acc_mean[3]*100
        ap_mean = np.mean(acc_mean)*100
        ap = ap_mean
            
        aps[method].append(ap)

# Convert lists to numpy arrays
for method in methods:
    aps[method] = np.array(aps[method], dtype=float)

print("Data loaded. Generating plot...")

# --- Plotting Section ---
fig, ax = plt.subplots(figsize=(15, 5))

# Define plotting parameters
colors = {methods[0]: 'red', methods[1]: 'blue'}

# --- Calculate Smoothed Data ---
x_values = np.arange(len(scene_id_img_id_pairs))
smoothed_data = {}

for method in methods:
    # Calculate smoothed data
    smoothed_data[method] = gaussian_filter1d(aps[method], sigma=sigma)
    
    # Plot raw and smoothed data
    ax.plot(smoothed_data[method], 
            label=f"{method_names[methods.index(method)]} (smoothed)", 
            linewidth=2, 
            color=colors[method])
    
    # ax.plot(aps[method], 
    #         label=f"{method_names[methods.index(method)]} (raw)", 
    #         linewidth=1, 
    #         linestyle='--', 
    #         color=colors[method], 
    #         alpha=0.5)

# --- Colorize the Gap Between Smoothed Lines ---
# Create boolean mask to handle NaNs safely
valid_data = ~np.isnan(smoothed_data[methods[0]]) & ~np.isnan(smoothed_data[methods[1]])

# Fill green where method[1] > method[0] (positive gap)
ax.fill_between(x_values, 
                smoothed_data[methods[0]], 
                smoothed_data[methods[1]],
                where=valid_data & (smoothed_data[methods[1]] >= smoothed_data[methods[0]]),
                facecolor='green', 
                alpha=0.3, 
                interpolate=True,
                label=f'Gap ({method_names[1]} > {method_names[0]})')

# Fill red where method[0] > method[1] (negative gap)
ax.fill_between(x_values, 
                smoothed_data[methods[0]], 
                smoothed_data[methods[1]],
                where=valid_data & (smoothed_data[methods[0]] > smoothed_data[methods[1]]),
                facecolor='red', 
                alpha=0.3, 
                interpolate=True,
                label=f'Gap ({method_names[0]} > {method_names[1]})')

# --- Final Plot Adjustments ---
ax.set_xlabel(f"Image Pair Index (First {len(x_values)} pairs from '{split}')")
ax.set_ylabel("Mean Accuracy (%)")
ax.set_title(f"Performance Comparison: {method_names[1]} vs {method_names[0]} ({split})")
ax.legend(loc='best')
ax.grid(True, linestyle='--', alpha=0.6)
# ax.set_ylim(0, 100)  # Set y-axis from 0-100% for better readability

plt.tight_layout()
output_filename = 'performance_comparison_with_gap.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.close(fig)

# --- Print Mean APs (Ignoring NaNs) ---
print("\nMean Performance (NaNs ignored):")
for method in methods:
    mean_ap = np.nanmean(aps[method])
    print(f'{method_names[methods.index(method)]}: {mean_ap:.2f}%')

print("\nScript finished.")
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import os
import pandas as pd
# Define the exact paths based on your file structure
paths = {
    # 'Beta-Space Exploration':'/Users/khushigarg/Downloads/impulsive_pda_onetenth/Results/beta_min=0.3,k=4,m=4,l=4,power=30/Beta_Space_Exp_SAC_mismatch_{}.npy',
    'Perfect CSI (SAC)': '/Users/khushigarg/Downloads/impulsive_pda_onetenth_50seeds/Results/beta_min=0.3,k=4,m=4,l=8,power=30/sac_mismatch_{}.npy',
    # 'Golden State (SAC)': '/Users/khushigarg/Downloads/impulsive_pda_onetenth/Results/beta_min=1.0,k=4,m=4,l=4,power=30/sac_golden_{}.npy'
}

num_seeds = 100

def load_and_average_scenario(base_path, num_seeds):
    """Load multiple seed files and compute mean and std"""
    all_seeds = []
    missing_files = []
    
    for seed in range(num_seeds):
        file_path = base_path.format(seed)
        if os.path.exists(file_path):
            data = np.load(file_path)
            all_seeds.append(data)
            print(f"✓ Loaded: {os.path.basename(file_path)}")
        else:
            missing_files.append(file_path)
            print(f"✗ Missing: {os.path.basename(file_path)}")
    
    if len(all_seeds) == 0:
        raise ValueError(f"No data files found for pattern: {base_path}")
    
    print(f"Loaded {len(all_seeds)}/{num_seeds} seed files")
    
    # Ensure all arrays have the same length (pad if necessary)
    max_length = max(len(arr) for arr in all_seeds)
    padded_seeds = []
    for arr in all_seeds:
        if len(arr) < max_length:
            # Pad with the last value
            padded = np.pad(arr, (0, max_length - len(arr)), mode='edge')
        else:
            padded = arr[:max_length]
        padded_seeds.append(padded)
    
    all_seeds = np.array(padded_seeds)
    mean_curve = np.mean(all_seeds, axis=0)
    std_curve = np.std(all_seeds, axis=0)
    
    return mean_curve, std_curve, len(all_seeds)

# Load and process all scenarios
print("Loading and processing scenarios...")
results = {}
for scenario, path_pattern in paths.items():
    print(f"\n--- Processing {scenario} ---")
    mean_curve, std_curve, num_loaded = load_and_average_scenario(path_pattern, num_seeds)
    results[scenario] = {
        'mean': mean_curve, 
        'std': std_curve,
        'num_seeds': num_loaded
    }

# Apply smoothing
window = 100 # Adjust for desired smoothness
print(f"\nApplying smoothing with window size: {window}")

for scenario in results:
    results[scenario]['mean_smooth'] = uniform_filter1d(results[scenario]['mean'], size=window)
    results[scenario]['std_smooth'] = uniform_filter1d(results[scenario]['std'], size=window)
for scenario in results:
    results[scenario]['mean_smooth'] = np.squeeze(results[scenario]['mean_smooth'])
    results[scenario]['std_smooth'] = np.squeeze(results[scenario]['std_smooth'])

# Create publication-quality plot
plt.figure(figsize=(10, 7))

# Color scheme and styles matching the paper
colors = {
    'Beta-Space Exploration': '#32CD32',  # Orange-red
    'Perfect CSI (SAC)': '#4169E1',       # Royal blue
    'Golden State (SAC)': '#FF4500'       # Lime green
}

linestyles = {
    'Beta-Space Exploration': '-',    # Solid
    'Perfect CSI (SAC)': '--',        # Dashed
    'Golden State (SAC)': '-'         # Solid
}

# Plot each scenario
for scenario in results:
    mean_smooth = results[scenario]['mean_smooth']
    std_smooth = results[scenario]['std_smooth']
    num_seeds_used = results[scenario]['num_seeds']
    
    steps = np.arange(len(mean_smooth))
    color = colors[scenario]
    linestyle = linestyles[scenario]
    
    # Plot mean curve
    plt.plot(steps, mean_smooth, 
             color=color, 
             linestyle=linestyle,
             linewidth=2.5, 
             label=f'{scenario} ({num_seeds_used} seeds)')
    
    # Add 95% confidence intervals
    se = std_smooth / np.sqrt(num_seeds_used)  # Standard error
    ci_95 = 1.96 * se  # 95% confidence interval
    
    plt.fill_between(steps, 
                     mean_smooth - ci_95, 
                     mean_smooth + ci_95,
                     color=color, 
                     alpha=0.15)

# Formatting to match paper style
plt.xlabel('Total Time Steps (×10³)', fontsize=14)
plt.ylabel('Sum Rate Rs (bps/Hz)', fontsize=14)
plt.title('(a) βmin = 0.3, Pt = 30 dBm,\nK = 4, M = 4, L = 16', fontsize=12)

# Set x-axis to show steps in thousands
max_steps = max(len(results[scenario]['mean_smooth']) for scenario in results)
x_ticks = np.arange(0, max_steps + 1, 4000)
x_labels = [f'{int(x/1000)}' for x in x_ticks]
plt.xticks(x_ticks, x_labels)

plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.2)
plt.xlim(0, 20000)
plt.ylim(0, 10)  # Adjust based on your data range

plt.tight_layout()
plt.savefig('RIS_Learning_Curves_Paper_Style.png', dpi=300, bbox_inches='tight')
plt.show()

# Print final performance summary
print("\n" + "="*60)
print("FINAL PERFORMANCE SUMMARY")
print("="*60)

for scenario in results:
    mean_curve = results[scenario]['mean']
    final_performance = np.mean(mean_curve[-1000:])  # Last 1000 steps
    final_std = np.std(mean_curve[-1000:])
    num_seeds_used = results[scenario]['num_seeds']
    
    print(f"{scenario:<25}: {final_performance:.2f} ± {final_std:.2f} bps/Hz ({num_seeds_used} seeds)")

# Performance comparison table
print("\n" + "="*60)
print("PERFORMANCE COMPARISON TABLE")
print("="*60)

# Calculate relative performance
golden_final = np.mean(results['Golden State (SAC)']['mean'][-1000:])

comparison_data = []
for scenario in results:
    mean_curve = results[scenario]['mean']
    final_perf = np.mean(mean_curve[-1000:])
    final_std = np.std(mean_curve[-1000:])
    
    # Calculate convergence step (95% of final performance)
    convergence_threshold = 0.95 * final_perf
    convergence_step = np.argmax(mean_curve > convergence_threshold)
    
    # Relative performance vs Golden State
    relative_perf = (final_perf / golden_final) * 100
    
    comparison_data.append({
        'Scenario': scenario,
        'Final Sum-Rate': f"{final_perf:.2f} ± {final_std:.2f}",
        'Convergence Step': convergence_step,
        'Relative Performance': f"{relative_perf:.1f}%",
        'Seeds': results[scenario]['num_seeds']
    })

# Print table

df = pd.DataFrame(comparison_data)
print(df.to_string(index=False))

print(f"\nPlot saved as: 'RIS_Learning_Curves_Paper_Style.png'")
print(f"Smoothing window used: {window} steps")

import matplotlib.pyplot as plt

# Exact Block Sizes from Table 2.1
block_sizes = [16, 32, 64, 128, 256] # 

# Miss Rates by Cache Size (Data from Table 2.1)
miss_rates_by_cache = {
    "2K":  [6.2, 3.1, 2.0, 0.7, 0.4], # 
    "4K":  [6.2, 3.1, 2.0, 0.7, 0.4], # 
    "8K":  [6.2, 3.1, 2.0, 0.7, 0.3], # 
    "16K": [6.2, 3.1, 1.5, 0.7, 0.3], # 
    "32K": [6.2, 3.1, 1.5, 0.7, 0.3], # 
}

plt.figure(figsize=(8, 5))

# Plotting lines for each Cache Size
for cache_size, rates in miss_rates_by_cache.items():
    plt.plot(block_sizes, rates, marker="o", linewidth=2, label=f"Cache {cache_size}")

# Formatting the plot for your report
plt.xlabel("Block Size (B)")
plt.ylabel("Miss Rate (%)")
plt.title("Matrix 128x128: Miss Rate vs Block Size (by Cache Size)")
plt.xticks(block_sizes) # Forces the X-axis to show the exact block numbers
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Cache Capacity (C)")
plt.tight_layout()

plt.show()
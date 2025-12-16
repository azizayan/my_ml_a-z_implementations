import matplotlib.pyplot as plt

cache_sizes = ["16", "32", "64", "128", "256"]
miss_rates_by_block = {
    "2K":  [6.2, 3.1, 2.0, 0.7, 0.4],
    "4K":  [6.2, 3.1, 2.0, 0.7, 0.4],
    "8K":  [6.2, 3.1, 2.0, 0.7, 0.3],
    "16K": [6.2, 3.1, 1.5, 0.7, 0.3],
    "32K": [6.2, 3.1, 1.5, 0.7, 0.3],
}

plt.figure(figsize=(6, 4))
for block_size, rates in miss_rates_by_block.items():
    plt.plot(cache_sizes, rates, marker="o", label=f"Block {block_size}")

plt.xlabel("Cache Size")
plt.ylabel("Miss Rate (%)")
plt.title("Miss Rate vs Cache Size (by Block Size)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(title="Block Size")
plt.tight_layout()
plt.show()
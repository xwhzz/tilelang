from tabulate import tabulate


table = [
    ["original", 1, 1, 1, 1],
    ["current", 1, 1, 1, 1]
]

headers = ["version", "Best Latency (s)", "Best TFlops", "Reference TFlops", "Best Config"]

print(tabulate(table, headers=headers, tablefmt="github", stralign="left", numalign="decimal"))
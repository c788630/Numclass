from numclass import classify_number

min_labels = None
max_labels = None
min_examples = []
max_examples = []

for n in range(1, 100001):
    result = classify_number(n)
    num_labels = len(result["classes"])

    # New minimum found!
    if (min_labels is None) or (num_labels < min_labels):
        min_labels = num_labels
        min_examples = [n]
        print(f"NEW MIN: n={n}, labels={num_labels}")
    elif num_labels == min_labels:
        min_examples.append(n)
        print(f"EQUAL MIN: n={n}, labels={num_labels}")

    # New maximum found!
    if (max_labels is None) or (num_labels > max_labels):
        max_labels = num_labels
        max_examples = [n]
        print(f"NEW MAX: n={n}, labels={num_labels}")
    elif num_labels == max_labels:
        max_examples.append(n)
        print(f"EQUAL MAX: n={n}, labels={num_labels}")

    if n % 100 == 0:
        print(f"Checked up to {n}...")

print("\n=== MINIMAL classification ===")
print(f"Fewest labels: {min_labels}")
print(f"Examples: {min_examples[:10]}{' ...' if len(min_examples) > 10 else ''}")

print("\n=== MAXIMAL classification ===")
print(f"Most labels: {max_labels}")
print(f"Examples: {max_examples[:10]}{' ...' if len(max_examples) > 10 else ''}")

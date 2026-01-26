from datasets import load_dataset

# 1. Load the dataset (replace with your target dataset)
# Using 'split="train"' ensures you get a Dataset object, not a DatasetDict
ds = load_dataset("compling/somo", split="train")

shuffled_ds = ds.shuffle(seed=42)
subset = shuffled_ds.select(range(15000))

print(f"Original size: {len(ds)}")
print(f"Subset size: {len(subset)}")

# 3. Upload to Hugging Face
# Replace 'your_username/new_subset_name' with your desired repo ID
subset.push_to_hub("compling/somo-15k-subset")

from pathlib import Path


dataset = Path("data/raw")

files = list(dataset.rglob("*.md"))

print("Total markdown files:", len(files))

for file in files[:10]:
    print(file)
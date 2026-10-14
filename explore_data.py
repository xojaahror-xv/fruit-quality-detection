from pathlib import Path
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

root = Path("data")
train_dir = next(root.rglob("train"))
test_dir = next(root.rglob("test"))
exts = {".jpg", ".jpeg", ".png"}

for split_dir in (train_dir, test_dir):
    print("\n", split_dir)
    total = 0
    for cls in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        n = len([f for f in cls.iterdir() if f.suffix.lower() in exts])
        print(f"  {cls.name:20s} {n}")
        total += n
    print("  JAMI:", total)

classes = sorted(p for p in train_dir.iterdir() if p.is_dir())
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for ax, cls in zip(axes.flat, classes):
    files = [f for f in cls.iterdir() if f.suffix.lower() in exts]
    img = Image.open(random.choice(files))
    ax.imshow(img)
    ax.set_title(f"{cls.name}\n{img.size}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("samples.png", dpi=100)
print("\nsamples.png saqlandi")

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

IMG = (224, 224)
BATCH = 32

test_dir = next(Path("data").rglob("test"))
class_names = json.load(open("class_names.json", encoding="utf-8"))

test_ds = keras.utils.image_dataset_from_directory(
    test_dir, image_size=IMG, batch_size=BATCH,
    label_mode="categorical", shuffle=False, class_names=class_names)

model = keras.models.load_model("fruit_model.keras")

y_true = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds, verbose=1), axis=1)

acc = (y_true == y_pred).mean()
print(f"\nTEST ANIQLIGI: {acc:.4f}\n")
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Bashorat"); plt.ylabel("Haqiqiy")
plt.title("Confusion matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
print("confusion_matrix.png saqlandi")

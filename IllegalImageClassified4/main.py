from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = YOLO("model/bestYOLOillegalImageClassified4.pt")

# Load image
img_path = "test/Screenshot 2025-07-14 213841.png"
img = Image.open(img_path)

# Run prediction
results = model(img)

# Get prediction info
probs = results[0].probs
top1 = probs.top1
confidence = probs.data[top1].item()
class_name = results[0].names[top1]

# Print result
print(f"🧠 Prediction: {class_name} ({confidence * 100:.2f}%)")

# Show image
plt.imshow(img)
plt.axis("off")
plt.title(f"Prediction: {class_name} ({confidence * 100:.2f}%)")
plt.show()
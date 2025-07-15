from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = YOLO("model/best.pt")

# Load image
img_path = "test/Slip/1a7d770a-07a5-46ed-8620-74af4862b4c5.jpg"
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
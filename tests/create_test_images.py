#!/usr/bin/env python3
"""Create simple test images for testing AI deduplication."""

from PIL import Image, ImageDraw
import os

# Create test_images directory if it doesn't exist
os.makedirs("test_images", exist_ok=True)

# Create some simple test images
colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
]

# Create base images
for i, color in enumerate(colors):
    img = Image.new("RGB", (224, 224), color)
    draw = ImageDraw.Draw(img)

    # Add some text to make them slightly different
    draw.text((50, 100), f"Image {i + 1}", fill=(255, 255, 255))
    img.save(f"test_images/image_{i + 1}.jpg")

# Create some similar images (slight variations)
# Red variant
img = Image.new("RGB", (224, 224), (250, 5, 5))  # Slightly different red
draw = ImageDraw.Draw(img)
draw.text((50, 100), "Image 1", fill=(255, 255, 255))
img.save("test_images/image_1_variant.jpg")

# Green variant
img = Image.new("RGB", (224, 224), (5, 250, 5))  # Slightly different green
draw = ImageDraw.Draw(img)
draw.text((50, 100), "Image 2", fill=(255, 255, 255))
img.save("test_images/image_2_variant.jpg")

print("Created test images successfully!")
print("Files created:")
for f in sorted(os.listdir("test_images")):
    if f.endswith(".jpg"):
        print(f"  {f}")

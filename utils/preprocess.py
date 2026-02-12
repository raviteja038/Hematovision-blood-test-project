from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


def preprocess_image(img_path):
    try:
        # Open image safely
        img = Image.open(img_path).convert("RGB")

        # Resize to model size
        img = img.resize((224, 224))

        # Convert to array
        arr = np.array(img, dtype=np.float32)

        # Normalize (0–1)
        arr = arr / 255.0

        # Add batch dimension
        arr = np.expand_dims(arr, axis=0)

        return arr

    except Exception as e:
        print("Preprocess Error:", e)
        return None

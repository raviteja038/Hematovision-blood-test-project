import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image

classes = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
model = None   # lazy load


def predict_cell(path):
    global model

    # Load model only once
    if model is None:
        model = load_model("BloodCell.h5")

    try:
        img = preprocess_image(path)

        pred = model.predict(img)
        confidence = float(np.max(pred))   # highest probability
        label_index = np.argmax(pred)

        # ---- INVALID IMAGE CHECK ----
        if confidence < 0.60:   # you can change 0.60 → 0.70 if needed
            return "INVALID"

        return classes[label_index]

    except Exception as e:
        print("Prediction Error:", e)
        return "INVALID"

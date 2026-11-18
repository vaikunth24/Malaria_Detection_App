

import tensorflow as tf
import os

MODEL_PATHS = {
    "Original MobileNetV2": "models/malaria_original.keras",
    "Threshold 100": "models/malaria_threshold100.keras",
    "Threshold 200": "models/malaria_threshold200.keras"
}

loaded_models = {}

print("TensorFlow:", tf.__version__)
print("\nStarting model loading test...\n")

for name, path in MODEL_PATHS.items():
    print(f"Loading {name}...")

    if not os.path.exists(path):
        print(f"FILE NOT FOUND: {path}\n")
        continue

    try:
        model = tf.keras.models.load_model(
            path,
            compile=False
        )

        loaded_models[name] = model

        print("SUCCESS")
        print("Input shape:", model.input_shape)
        print("Output shape:", model.output_shape)
        print("-" * 40)

    except Exception as e:
        print("FAILED:", repr(e))
        print("-" * 40)

print(
    f"\nSuccessfully loaded: {len(loaded_models)} out of 3 models"
)


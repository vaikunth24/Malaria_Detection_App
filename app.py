import csv
import io
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Malaria Parasite Detection",
    page_icon="🔬",
    layout="wide"
)
# =====================================================
# PREDICTION HISTORY INITIALIZATION
# =====================================================

if "analysis_completed" not in st.session_state:
    st.session_state.analysis_completed = False
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

st.title("🔬 Malaria Parasite Detection System")

st.markdown(
    """
    **AI-powered microscopic blood cell classification**

    Using MobileNetV2 and Selective Intensity Ensemble Classifier (SIEC).
    """
)

st.divider()


# =====================================================
# 2. LOAD TRAINED MODELS
# =====================================================

@st.cache_resource
def load_models():

    original_model = tf.keras.models.load_model(
        "models/malaria_original.keras",
        compile=False
    )

    threshold100_model = tf.keras.models.load_model(
        "models/malaria_threshold100.keras",
        compile=False
    )

    threshold200_model = tf.keras.models.load_model(
        "models/malaria_threshold200.keras",
        compile=False
    )

    return (
        original_model,
        threshold100_model,
        threshold200_model
    )


try:

    with st.spinner("Loading trained models..."):

        original_model, model100, model200 = load_models()

    st.success("All three trained models loaded successfully!")

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()


# =====================================================
# 3. IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image, threshold=None):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    ).copy()

    # Apply intensity thresholding
    if threshold is not None:

        intensity = np.max(
            image_array,
            axis=-1
        )

        image_array[intensity < threshold] = 0

    # MobileNetV2 preprocessing
    image_array = preprocess_input(
        image_array
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =====================================================
# 4. ORIGINAL MOBILENETV2 PREDICTION
# =====================================================

def predict_original(image):

    image_input = preprocess_image(image)

    probability = original_model.predict(
        image_input,
        verbose=0
    )[0][0]

    return float(probability)


# =====================================================
# 5. WEIGHTED SIEC PREDICTION
# =====================================================

def predict_siec(image):

    # Original RGB branch
    input_original = preprocess_image(image)

    # Threshold 100 branch
    input100 = preprocess_image(
        image,
        threshold=100
    )

    # Threshold 200 branch
    input200 = preprocess_image(
        image,
        threshold=200
    )

    # Individual model predictions
    p_original = original_model.predict(
        input_original,
        verbose=0
    )[0][0]

    p100 = model100.predict(
        input100,
        verbose=0
    )[0][0]

    p200 = model200.predict(
        input200,
        verbose=0
    )[0][0]

    # Validation-optimized weights
    weighted_probability = (
        0.90 * p_original +
        0.05 * p100 +
        0.05 * p200
    )

    return float(weighted_probability)


# =====================================================
# 6. CLASSIFICATION RESULT
# =====================================================

def get_prediction(probability):

    probability = float(
        np.clip(probability, 0.0, 1.0)
    )

    if probability >= 0.5:

        prediction = "Parasitized"
        confidence = probability * 100

    else:

        prediction = "Uninfected"
        confidence = (1 - probability) * 100

    return prediction, confidence



def record_prediction(
    image_name,
    model_name,
    prediction,
    confidence,
    probability
):

    from datetime import datetime

    history_entry = {
        "Date & Time": datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        ),
        "Image Name": image_name,
        "Model": model_name,
        "Prediction": prediction,
        "Confidence": f"{confidence:.2f}%",
        "Parasitized Probability": f"{probability * 100:.2f}%"
    }

    st.session_state.prediction_history.append(
        history_entry
    )


# =====================================================
# 7. GRAD-CAM GENERATION
# Original MobileNetV2
# =====================================================

def generate_gradcam(image):

    base_model = original_model.get_layer(
        "mobilenetv2_1.00_224"
    )

    last_conv_layer = base_model.get_layer(
        "Conv_1"
    )

    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    gap_layer = original_model.get_layer(
        "global_average_pooling2d"
    )

    dropout_layer = original_model.get_layer(
        "dropout"
    )

    dense_layer = original_model.get_layer(
        "dense"
    )

    image_input = preprocess_image(image)

    image_tensor = tf.convert_to_tensor(
        image_input,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs = feature_extractor(
            image_tensor,
            training=False
        )

        tape.watch(conv_outputs)

        x = gap_layer(conv_outputs)

        x = dropout_layer(
            x,
            training=False
        )

        predictions = dense_layer(x)

        probability = predictions[:, 0]

        # Explain the predicted class
        target_score = tf.where(
            probability >= 0.5,
            probability,
            1.0 - probability
        )

    gradients = tape.gradient(
        target_score,
        conv_outputs
    )

    if gradients is None:
        raise ValueError(
            "Grad-CAM gradients could not be calculated."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


# =====================================================
# 8. GENERATE JET HEATMAP USING NUMPY
# =====================================================

def create_jet_heatmap(heatmap):

    heatmap = np.clip(
        heatmap,
        0,
        1
    )

    red = np.clip(
        1.5 - np.abs(4 * heatmap - 3),
        0,
        1
    )

    green = np.clip(
        1.5 - np.abs(4 * heatmap - 2),
        0,
        1
    )

    blue = np.clip(
        1.5 - np.abs(4 * heatmap - 1),
        0,
        1
    )

    colored_heatmap = np.stack(
        [red, green, blue],
        axis=-1
    )

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    return Image.fromarray(
        colored_heatmap
    )


# =====================================================
# 9. CREATE GRAD-CAM OVERLAY
# =====================================================

def create_gradcam_overlay(image, heatmap):

    original_image = image.convert(
        "RGB"
    ).resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    heatmap_image = create_jet_heatmap(
        heatmap
    )

    heatmap_image = heatmap_image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    original_array = np.asarray(
        original_image,
        dtype=np.float32
    )

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32
    )

    overlay_array = (
        0.6 * original_array +
        0.4 * heatmap_array
    )

    overlay_array = np.clip(
        overlay_array,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        overlay_array
    )


# =====================================================
# 10. MODEL SELECTION
# =====================================================

st.subheader("Choose Prediction Model")

selected_model = st.selectbox(
    "Select a prediction mode",
    [
        "Original MobileNetV2",
        "Optimized Weighted SIEC",
        "Compare Both Models"
    ]
)


# =====================================================
# 11. IMAGE UPLOAD
# =====================================================

st.divider()

st.subheader("Upload Microscopic Blood Cell Image")

uploaded_file = st.file_uploader(
    "Choose a blood cell image",
    type=["jpg", "jpeg", "png"],
    key="malaria_image_uploader"
)


# =====================================================
# IMAGE QUALITY CHECK
# =====================================================

if uploaded_file is not None:

    st.subheader("🖼️ Image Quality Assessment")

    # Read uploaded image
    quality_image = Image.open(uploaded_file).convert("RGB")

    # Image dimensions
    width, height = quality_image.size

    # Convert to NumPy array
    image_array = np.array(quality_image)

    # Calculate average brightness
    brightness = np.mean(image_array)

    # Display image information
    col1, col2, col3 = st.columns(3)

    col1.metric("Image Width", f"{width} px")
    col2.metric("Image Height", f"{height} px")
    col3.metric("Average Brightness", f"{brightness:.2f}")

    # Quality assessment
    if width < 224 or height < 224:

        st.warning(
            "⚠️ Low-resolution image detected. "
            "Prediction reliability may be affected."
        )

    elif brightness < 50:

        st.warning(
            "⚠️ The image appears very dark. "
            "Consider uploading a better-lit image."
        )

    elif brightness > 220:

        st.warning(
            "⚠️ The image appears very bright or overexposed."
        )

    else:

        st.success(
            "✅ Image brightness and resolution "
            "appear acceptable for analysis."
        )


if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as e:

        st.error(
            f"Unable to read uploaded image: {e}"
        )

        st.stop()


    # Image preview and information
    image_col, info_col = st.columns(2)

    with image_col:

        st.markdown("### Uploaded Image")

        st.image(
            image,
            caption="Microscopic Blood Cell Image",
            width=350
        )

    with info_col:

        st.markdown("### Image Information")

        st.write(
            "Image dimensions:",
            image.size
        )

        st.write(
            "Model input size:",
            "224 × 224"
        )

        st.write(
            "Color channels:",
            "RGB"
        )


    st.divider()


    # =================================================
    # 12. ANALYZE IMAGE
    # =================================================

    if st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True
    ):
       

        # ---------------------------------------------
        # PREDICTION
        # ---------------------------------------------

        try:

            with st.spinner(
                "Analyzing blood cell image..."
            ):

                original_probability = predict_original(
                    image
                )

                siec_probability = predict_siec(
                    image
                )

                original_prediction, original_confidence = (
                    get_prediction(original_probability)
                )

                siec_prediction, siec_confidence = (
                    get_prediction(siec_probability)
                )
                st.session_state.analysis_completed = True

                if selected_model == "Original MobileNetV2":

                    record_prediction(
                    uploaded_file.name,
                    "Original MobileNetV2",
                    original_prediction,
                    original_confidence,
                    original_probability
                )

                elif selected_model == "Optimized Weighted SIEC":

                    record_prediction(
                    uploaded_file.name,
                    "Optimized Weighted SIEC",
                    siec_prediction,
                    siec_confidence,
                    siec_probability
                )

                else:

                     # Record both model predictions
                    record_prediction(
                    uploaded_file.name,
                    "Original MobileNetV2",
                    original_prediction,
                    original_confidence,
                    original_probability
                )

                record_prediction(
                uploaded_file.name,
                "Optimized Weighted SIEC",
                siec_prediction,
                siec_confidence,
                siec_probability
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

            st.stop()


        # =============================================
        # 13. PREDICTION RESULTS
        # =============================================

        st.divider()

        st.header("Prediction Results")

        if selected_model == "Original MobileNetV2":

            results = [
                (
                    "Original MobileNetV2",
                    original_prediction,
                    original_confidence,
                    original_probability
                )
            ]

        elif selected_model == "Optimized Weighted SIEC":

            results = [
                (
                    "Optimized Weighted SIEC",
                    siec_prediction,
                    siec_confidence,
                    siec_probability
                )
            ]

        else:

            results = [
                (
                    "Original MobileNetV2",
                    original_prediction,
                    original_confidence,
                    original_probability
                ),
                (
                    "Optimized Weighted SIEC",
                    siec_prediction,
                    siec_confidence,
                    siec_probability
                )
            ]


        result_columns = st.columns(
            len(results)
        )

        for column, result in zip(
            result_columns,
            results
        ):

            model_name, prediction, confidence, probability = result

            with column:

                st.subheader(model_name)

                st.metric(
                    "Predicted Class",
                    prediction
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.write(
                    f"Parasitized probability: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(confidence / 100)
                )


        # =============================================
        # 14. MODEL AGREEMENT AND COMPARISON
        # =============================================

        if selected_model == "Compare Both Models":

            st.divider()

            st.header("Model Agreement")

            if original_prediction == siec_prediction:

                st.success(
                    f"Both models agree: {original_prediction}"
                )

            else:

                st.warning(
                    "The models produced different predictions."
                )


            st.subheader("Detailed Model Comparison")

            compare_col1, compare_col2 = st.columns(2)

            with compare_col1:

                st.markdown("### Original MobileNetV2")

                st.metric(
                    "Predicted Class",
                    original_prediction
                )

                st.metric(
                    "Confidence",
                    f"{original_confidence:.2f}%"
                )

                st.metric(
                    "Parasitized Probability",
                    f"{original_probability * 100:.2f}%"
                )

            with compare_col2:

                st.markdown("### Optimized Weighted SIEC")

                st.metric(
                    "Predicted Class",
                    siec_prediction
                )

                st.metric(
                    "Confidence",
                    f"{siec_confidence:.2f}%"
                )

                st.metric(
                    "Parasitized Probability",
                    f"{siec_probability * 100:.2f}%"
                )


        # =============================================
        # 15. GRAD-CAM VISUALIZATION
        # =============================================

        st.divider()

        st.header("Explainable AI - Grad-CAM")

        st.write(
            "Grad-CAM highlights image regions that influence "
            "the Original MobileNetV2 classification."
        )

        try:

            with st.spinner(
                "Generating Grad-CAM visualization..."
            ):

                heatmap = generate_gradcam(
                    image
                )

                overlay = create_gradcam_overlay(
                    image,
                    heatmap
                )

                original_display = image.resize(
                    (224, 224),
                    Image.Resampling.BILINEAR
                )

            gradcam_col1, gradcam_col2 = st.columns(2)

            with gradcam_col1:

                st.image(
                    original_display,
                    caption="Original Image",
                    use_container_width=True
                )

            with gradcam_col2:

                st.image(
                    overlay,
                    caption="Grad-CAM Overlay",
                    use_container_width=True
                )

            st.caption(
                "Grad-CAM explains influential image regions "
                "for Original MobileNetV2. It is not a validated "
                "parasite segmentation mask."
            )

        except Exception as e:

            st.warning(
                "Prediction completed successfully, but "
                "Grad-CAM could not be generated."
            )

            st.error(
                f"Grad-CAM error: {e}"
            )


        # =============================================
        # 16. RESEARCH DISCLAIMER
        # =============================================

        st.divider()

        st.info(
            "Research demonstration only. This application "
            "is not intended for clinical diagnosis. Predictions "
            "should not replace professional laboratory examination."
        )


# =====================================================
# 17. FOOTER
# =====================================================

st.divider()

st.caption(
    "Malaria Parasite Detection | MobileNetV2 | "
    "Weighted SIEC | Explainable AI"
)

st.caption(
    "Academic Research Project"
)

# =====================================================
# PREDICTION HISTORY
# =====================================================

# =====================================================
# PREDICTION HISTORY
# =====================================================

st.divider()

st.header("📜 Prediction History")

history = st.session_state.prediction_history

if history:

    st.caption(f"Total recorded predictions: {len(history)}")

    # Display history
    for entry in reversed(history):

        with st.expander(
            f"{entry['Date & Time']} | "
            f"{entry['Prediction']} | "
            f"{entry['Image Name']}"
        ):

            col1, col2, col3 = st.columns(3)

            col1.write(f"**Model:** {entry['Model']}")

            col2.write(
                f"**Confidence:** {entry['Confidence']}"
            )

            col3.write(
                f"**Parasitized Probability:** "
                f"{entry['Parasitized Probability']}"
            )

    # Download prediction history as CSV
    st.subheader("📥 Export History")

    csv_buffer = io.StringIO()

    fieldnames = [
        "Date & Time",
        "Image Name",
        "Model",
        "Prediction",
        "Confidence",
        "Parasitized Probability"
    ]

    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(history)

    st.download_button(
        label="📥 Download Prediction History (CSV)",
        data=csv_buffer.getvalue(),
        file_name="malaria_prediction_history.csv",
        mime="text/csv"
    )

    # Clear history
    st.divider()

    if st.button("🗑️ Clear Prediction History"):

        st.session_state.prediction_history = []

        st.rerun()

else:

    st.info(
        "No predictions recorded yet. "
        "Analyze an image to populate your history."
    )

# =====================================================
# MODEL PERFORMANCE DASHBOARD
# =====================================================

if st.session_state.analysis_completed:

    st.divider()

    st.header("📊 Model Performance Dashboard")

    st.caption(
        "Test-set performance of the malaria detection models"
    )

    # =================================================
    # ORIGINAL MOBILENETV2
    # =================================================

    st.subheader("Original MobileNetV2")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", "95.75%")
    col2.metric("Precision", "96.17%")
    col3.metric("Recall", "95.30%")
    col4.metric("F1 Score", "95.73%")
    col5.metric("ROC-AUC", "98.64%")

    # =================================================
    # OPTIMIZED WEIGHTED SIEC
    # =================================================

    st.subheader("Optimized Weighted SIEC")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", "95.70%")
    col2.metric("Precision", "96.07%")
    col3.metric("Recall", "95.30%")
    col4.metric("F1 Score", "95.68%")
    col5.metric("ROC-AUC", "98.65%")

    # =================================================
    # PERFORMANCE COMPARISON
    # =================================================

    st.subheader("🔍 Performance Comparison")

    st.info(
        """
        **Research Observation:**

        The Original MobileNetV2 model achieved slightly higher
        Accuracy, Precision, and F1 Score.

        The Optimized Weighted SIEC achieved a slightly higher
        ROC-AUC score.

        Overall, the Weighted SIEC approach maintained performance
        very close to the original model while combining predictions
        from three preprocessing branches.
        """
    )


  

# =====================================================
# ROBUSTNESS TESTING DASHBOARD
# =====================================================

if st.session_state.analysis_completed:

    st.divider()

    st.header("🛡️ Robustness Testing Dashboard")

    st.caption(
        "Performance under different image-quality conditions"
    )

    # =================================================
    # BRIGHTNESS ROBUSTNESS
    # =================================================

    st.subheader("☀️ Brightness Variation")

    col1, col2 = st.columns(2)

    col1.metric(
        "Original MobileNetV2",
        "94.05%"
    )

    col2.metric(
        "Optimized Weighted SIEC",
        "94.20%",
        delta="+0.15%"
    )

    # =================================================
    # NOISE ROBUSTNESS
    # =================================================

    st.subheader("🔊 Noise")

    col1, col2 = st.columns(2)

    col1.metric(
        "Original MobileNetV2",
        "74.60%"
    )

    col2.metric(
        "Optimized Weighted SIEC",
        "74.25%",
        delta="-0.35%"
    )

    st.warning(
        "Noise significantly affects both models. "
        "The infected-class recall also decreases under noisy conditions, "
        "indicating that noisy images require careful interpretation."
    )

    # =================================================
    # BLUR ROBUSTNESS
    # =================================================

    st.subheader("🌫️ Blur")

    col1, col2 = st.columns(2)

    col1.metric(
        "Original MobileNetV2",
        "91.45%"
    )

    col2.metric(
        "Optimized Weighted SIEC",
        "91.45%",
        delta="0.00%"
    )

    # =================================================
    # ROBUSTNESS SUMMARY
    # =================================================

    st.subheader("📋 Robustness Summary")

    st.info(
        """
        **Key observations:**

        • Weighted SIEC performs slightly better under brightness variation.

        • Original MobileNetV2 performs slightly better under noise.

        • Both models achieve the same accuracy under blur.

        • Noise is the most challenging image-quality condition
          among the evaluated robustness tests.
        """
    )
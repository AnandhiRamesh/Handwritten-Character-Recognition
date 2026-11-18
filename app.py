
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageEnhance
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# PAGE CONFIGURATION


st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS - LIGHT AND DARK THEME SUPPORT


st.markdown("""
<style>

.stApp {
    background: var(--background-color);
    color: var(--text-color);
}

[data-testid="stSidebar"] {
    background: var(--secondary-background-color);
    border-right: 1px solid rgba(128,128,128,0.2);
}

[data-testid="stSidebar"] * {
    color: var(--text-color);
}

.main-title {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-color);
    margin-bottom: 5px;
}

.subtitle {
    color: var(--text-color);
    opacity: 0.7;
    font-size: 15px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 23px;
    font-weight: 650;
    margin-top: 20px;
    margin-bottom: 15px;
}

.metric-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    padding: 22px;
    border-radius: 12px;
    min-height: 115px;
}

.metric-label {
    font-size: 14px;
    opacity: 0.75;
}

.metric-value {
    font-size: 27px;
    font-weight: 700;
    margin-top: 8px;
}

.result-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.25);
    padding: 25px;
    border-radius: 14px;
    text-align: center;
}

.result-digit {
    font-size: 65px;
    font-weight: 750;
    margin: 10px;
}

div.stButton > button {
    width: 100%;
    border-radius: 8px;
    border: 1px solid rgba(128,128,128,0.25);
    padding: 10px;
    font-weight: 600;
    background: var(--secondary-background-color);
    color: var(--text-color);
}

div.stButton > button:hover {
    border-color: var(--primary-color);
    color: var(--primary-color);
}

hr {
    border-color: rgba(128,128,128,0.2);
}

</style>
""", unsafe_allow_html=True)


# MODEL LOADING

MODEL_PATH = Path(__file__).parent / "handwritten_model.keras"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "handwritten_model.keras was not found."
        )

    return tf.keras.models.load_model(MODEL_PATH)

# SESSION STATE

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "history" not in st.session_state:
    st.session_state.history = []


# IMAGE PREPROCESSING

def preprocess_image(image):

    # Convert image to grayscale
    image = ImageOps.grayscale(image)

    # Improve contrast
    image = ImageEnhance.Contrast(image).enhance(2.0)

    # Convert to NumPy array
    img_array = np.array(image)

    # Automatically invert white-background images
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Remove background noise
    img_array = np.where(img_array > 40, img_array, 0).astype(
        np.uint8
    )

    # Find digit boundaries
    coords = np.argwhere(img_array > 0)

    if coords.size == 0:
        raise ValueError(
            "No handwritten digit detected. Please upload a clear image."
        )

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img_array[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # Preserve aspect ratio and resize
    digit = Image.fromarray(cropped)
    digit.thumbnail((20, 20))

    # Create 28 x 28 black canvas
    canvas = Image.new("L", (28, 28), 0)

    # Center the digit
    x_position = (28 - digit.width) // 2
    y_position = (28 - digit.height) // 2

    canvas.paste(digit, (x_position, y_position))

    # Normalize pixel values
    final_image = np.array(canvas).astype("float32") / 255.0

    # Add dimensions for CNN
    final_image = final_image.reshape(1, 28, 28, 1)

    return final_image, canvas


# SIDEBAR NAVIGATION

with st.sidebar:

    st.markdown("## Handwritten AI")

    st.caption("Character Recognition System")

    st.divider()

    st.markdown("### Navigation")

    pages = [
        "Dashboard",
        "Digit Recognition",
        "Model Performance",
        "Prediction History",
        "About Project"
    ]

    for page in pages:

        if st.button(
            page,
            key="nav_" + page,
            use_container_width=True
        ):
            st.session_state.page = page
            st.rerun()

    st.divider()

    st.caption("Machine Learning Project")

    st.caption("CNN Based Digit Recognition")


# DASHBOARD


if st.session_state.page == "Dashboard":

    st.markdown(
        '<div class="main-title">Handwritten Character Recognition</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">An intelligent system to recognize handwritten digits using Convolutional Neural Networks.</div>',
        unsafe_allow_html=True
    )

    try:
        model = load_model()
        model_status = "Loaded"
    except Exception as e:
        model = None
        model_status = "Not Available"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Training Images</div>
            <div class="metric-value">60,000</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Digit Classes</div>
            <div class="metric-value">10</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Status</div>
            <div class="metric-value">{model_status}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This project uses a Convolutional Neural Network (CNN) "
        "to recognize handwritten digits from images."
    )

    st.write(
        "The system processes uploaded images, extracts digit "
        "information and predicts the corresponding digit "
        "from 0 to 9."
    )

    st.markdown(
        '<div class="section-title">Technology Stack</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    col1.info("Python\n\nProgramming Language")
    col2.info("TensorFlow / Keras\n\nDeep Learning")
    col3.info("Streamlit\n\nWeb Application")

    if model is None:
        st.warning(
            "Model file is missing or could not be loaded. "
            "Place handwritten_model.keras in the same folder as app.py."
        )


# DIGIT RECOGNITION

elif st.session_state.page == "Digit Recognition":

    st.markdown(
        '<div class="main-title">Digit Recognition</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Upload a handwritten digit image to get a prediction.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload Handwritten Digit Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        original_image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Uploaded Image")
            st.image(
                original_image,
                use_container_width=True
            )

        if st.button("Recognize Digit", type="primary"):

            try:
                model = load_model()

                processed_data, processed_image = preprocess_image(
                    original_image
                )

                predictions = model.predict(
                    processed_data,
                    verbose=0
                )

                predicted_digit = int(
                    np.argmax(predictions[0])
                )

                confidence = float(
                    np.max(predictions[0]) * 100
                )

                with col2:
                    st.markdown("### Processed Image")
                    st.image(
                        processed_image,
                        width=180
                    )

                st.markdown("---")

                st.markdown("### Prediction Result")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="metric-label">Predicted Digit</div>
                        <div class="result-digit">{predicted_digit}</div>
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{confidence:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(confidence / 100)

                # Store prediction history
                st.session_state.history.append({
                    "Date": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "Predicted Digit": predicted_digit,
                    "Confidence": round(confidence, 2)
                })

                st.markdown("### Class Probabilities")

                probability_df = pd.DataFrame({
                    "Digit": list(range(10)),
                    "Probability (%)": (
                        predictions[0] * 100
                    ).round(2)
                })

                st.bar_chart(
                    probability_df.set_index("Digit")
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        st.info(
            "Please upload a clear image containing one handwritten digit."
        )


# MODEL PERFORMANCE


elif st.session_state.page == "Model Performance":

    st.markdown(
        '<div class="main-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Evaluate the trained CNN model on MNIST test data.</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Click the button below to evaluate the saved model "
        "using the MNIST test dataset."
    )

    if st.button("Evaluate Model", type="primary"):

        try:
            with st.spinner("Evaluating model..."):

                model = load_model()

                (_, _), (x_test, y_test) = (
                    tf.keras.datasets.mnist.load_data()
                )

                x_test = x_test.astype("float32") / 255.0

                x_test = np.expand_dims(
                    x_test,
                    axis=-1
                )

                loss, accuracy = model.evaluate(
                    x_test,
                    y_test,
                    verbose=0
                )

                predictions = model.predict(
                    x_test,
                    verbose=0
                )

                predicted_labels = np.argmax(
                    predictions,
                    axis=1
                )

                from sklearn.metrics import (
                    classification_report,
                    confusion_matrix
                )

                report = classification_report(
                    y_test,
                    predicted_labels,
                    output_dict=True,
                    zero_division=0
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Test Accuracy",
                    f"{accuracy * 100:.2f}%"
                )

                col2.metric(
                    "Test Loss",
                    f"{loss:.4f}"
                )

                col3.metric(
                    "Test Images",
                    f"{len(y_test):,}"
                )

                st.markdown("### Classification Report")

                report_df = pd.DataFrame(report).transpose()

                st.dataframe(
                    report_df.round(4),
                    use_container_width=True
                )

                st.markdown("### Confusion Matrix")

                cm = confusion_matrix(
                    y_test,
                    predicted_labels
                )

                fig, ax = plt.subplots(figsize=(8, 6))

                image = ax.imshow(
                    cm,
                    cmap="Blues"
                )

                ax.set_title("CNN Confusion Matrix")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("Actual Label")

                ax.set_xticks(range(10))
                ax.set_yticks(range(10))

                fig.colorbar(image, ax=ax)

                st.pyplot(fig)
                plt.close(fig)

        except Exception as e:
            st.error(f"Evaluation failed: {e}")


# PREDICTION HISTORY

elif st.session_state.page == "Prediction History":

    st.markdown(
        '<div class="main-title">Prediction History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">View your previously recognized handwritten digits.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Prediction History",
            data=history_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info(
            "No prediction history available. "
            "Go to Digit Recognition and upload an image."
        )


# ABOUT PROJECT

elif st.session_state.page == "About Project":

    st.markdown(
        '<div class="main-title">About Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Project information and methodology.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Project Objective")

    st.write(
        "The main objective is to develop an intelligent "
        "handwritten digit recognition system using "
        "Convolutional Neural Networks."
    )

    st.markdown("### Dataset")

    st.write(
        "The project uses the MNIST handwritten digit dataset "
        "containing 60,000 training images and 10,000 testing images."
    )

    st.markdown("### CNN Architecture")

    st.write("""
    1. Input Layer
    2. Convolutional Layer
    3. Max Pooling Layer
    4. Second Convolutional Layer
    5. Second Max Pooling Layer
    6. Flatten Layer
    7. Dense Layer
    8. Dropout Layer
    9. Output Layer with 10 classes
    """)

    st.markdown("### Technologies Used")

    st.write("""
    - Python
    - TensorFlow
    - Keras
    - NumPy
    - Pandas
    - Matplotlib
    - Scikit-learn
    - Streamlit
    """)

    st.markdown("### Project Category")

    st.write(
        "Machine Learning Internship Project – "
        "Handwritten Character Recognition."
    )

    st.markdown("---")

    st.caption(
        "Handwritten Character Recognition | "
        "CNN Based Machine Learning Application"
    )

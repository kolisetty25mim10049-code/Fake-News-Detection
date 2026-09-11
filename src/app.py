import streamlit as st
import os
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #aaaaaa;
    margin-bottom: 35px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

.fake-box {
    background-color: #3b2025;
    border: 1px solid #ff4b4b;
}

.real-box {
    background-color: #173d2a;
    border: 1px solid #21c55d;
}

.result-title {
    font-size: 30px;
    font-weight: 700;
}

.info-box {
    background-color: #172d44;
    padding: 18px;
    border-radius: 12px;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #888888;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">📰 Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An AI & Machine Learning based system for detecting potentially fake news'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

BASE_PATH = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_PATH,
    "fake_news_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_PATH,
    "tfidf_vectorizer.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

except Exception as e:
    st.error("❌ Model files could not be loaded.")
    st.stop()


# =========================================================
# NEWS INPUT
# =========================================================

st.subheader("📝 Enter News Article")

news_text = st.text_area(
    "Paste your news article below:",
    height=220,
    placeholder="Example: Scientists announced a new discovery..."
)


# =========================================================
# BUTTONS
# =========================================================

col1, col2 = st.columns(2)

with col1:
    detect_button = st.button(
        "🔍 Detect News",
        use_container_width=True
    )

with col2:
    clear_button = st.button(
        "🧹 Clear",
        use_container_width=True
    )


if clear_button:
    st.rerun()


# =========================================================
# PREDICTION
# =========================================================

if detect_button:

    if not news_text.strip():

        st.warning("⚠️ Please enter a news article first.")

    else:

        # Convert text into TF-IDF features
        text_vector = vectorizer.transform([news_text])

        # Make prediction
        prediction = model.predict(text_vector)[0]

        # Get confidence
        probabilities = model.predict_proba(text_vector)[0]
        confidence = max(probabilities) * 100

        # Convert prediction to uppercase
        prediction = str(prediction).upper()


        # =================================================
        # RESULT
        # =================================================

        st.subheader("📊 Detection Result")

        if prediction == "FAKE":

            st.markdown(
                f"""
                <div class="result-box fake-box">
                    <div class="result-title">❌ FAKE NEWS</div>
                    <p>The model classified this article as potentially fake.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-box real-box">
                    <div class="result-title">✅ REAL NEWS</div>
                    <p>The model classified this article as likely real.</p>
                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # PREDICTION DETAILS
        # =================================================

        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Prediction",
                prediction
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


        # =================================================
        # CONFIDENCE BAR
        # =================================================

        st.write("### Confidence Level")

        st.progress(
            int(confidence)
        )


        # =================================================
        # EXPLANATION
        # =================================================

        if prediction == "FAKE":

            st.warning(
                "⚠️ The article contains patterns that are more "
                "similar to examples classified as fake in the training data."
            )

        else:

            st.success(
                "✅ The article contains patterns that are more "
                "similar to examples classified as real in the training data."
            )


        # =================================================
        # INFORMATION
        # =================================================

        st.markdown(
            """
            <div class="info-box">
            <b>ℹ️ How does it work?</b><br><br>
            1. The entered news article is processed as text.<br>
            2. TF-IDF converts the text into numerical features.<br>
            3. A Logistic Regression machine-learning model analyzes the features.<br>
            4. The system predicts whether the article is REAL or FAKE.<br>
            5. The prediction confidence is displayed to the user.
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# ABOUT PROJECT
# =========================================================

with st.expander("📚 About This Project"):

    st.write(
        """
        This project demonstrates the use of Machine Learning
        and Natural Language Processing (NLP) for fake news detection.

        **Machine Learning Model:** Logistic Regression

        **Feature Extraction:** TF-IDF Vectorizer

        **Interface:** Streamlit

        **Programming Language:** Python
        """
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.info(
    "⚠️ This project is an educational demonstration. "
    "The prediction does not guarantee that a news article is factually true or false."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Fake News Detection System | AI & Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
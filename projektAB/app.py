import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/best_model.joblib"
RESULTS_PATH = "reports/model_results.csv"


st.set_page_config(
    page_title="Predikcija zanra knjige",
    page_icon="📚",
    layout="centered"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_results():
    results = pd.read_csv(RESULTS_PATH)

    for column in ["Accuracy", "Macro F1", "Weighted F1"]:
        if column in results.columns:
            results[column] = results[column] * 100

    return results


st.title("📚 Predikcija zanra knjige na temelju opisa")

st.write(
    "Unesi opis knjige na engleskom jeziku, a sustav ce pokusati predvidjeti kojem zanru knjiga pripada."
)

try:
    model = load_model()
    results = load_results()

    best_by_macro_f1 = results.sort_values(by="Macro F1", ascending=False).iloc[0]
    best_by_accuracy = results.sort_values(by="Accuracy", ascending=False).iloc[0]

    used_model_name = best_by_macro_f1["Model"]

    st.subheader("Informacije o treniranim modelima")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Model koji aplikacija koristi",
            value=used_model_name,
            delta=f"Macro F1: {best_by_macro_f1['Macro F1']:.2f}%"
        )

    with col2:
        st.metric(
            label="Najtocniji model po Accuracy",
            value=best_by_accuracy["Model"],
            delta=f"Accuracy: {best_by_accuracy['Accuracy']:.2f}%"
        )

    with st.expander("Prikazi rezultate svih modela"):
        display_results = results.copy()
        display_results["Accuracy"] = display_results["Accuracy"].map(lambda x: f"{x:.2f}%")
        display_results["Macro F1"] = display_results["Macro F1"].map(lambda x: f"{x:.2f}%")
        display_results["Weighted F1"] = display_results["Weighted F1"].map(lambda x: f"{x:.2f}%")

        st.dataframe(display_results, use_container_width=True)

    st.markdown("---")

    book_description = st.text_area(
        "Opis knjige",
        height=180,
        placeholder="Primjer: A young wizard discovers ancient magic and must fight a dark sorcerer..."
    )

    if st.button("Predvidi zanr"):
        if not book_description.strip():
            st.warning("Molim unesi opis knjige.")
        else:
            prediction = model.predict([book_description])[0]

            st.success(f"Predvideni zanr: {prediction}")
            st.write(f"Model koristen za ovu predikciju: **{used_model_name}**")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([book_description])[0]
                classes = model.classes_

                top_predictions = sorted(
                    zip(classes, probabilities),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                st.subheader("Top 5 predikcija")

                for genre, probability in top_predictions:
                    st.write(f"{genre}: {probability:.2%}")
                    st.progress(float(probability))
            else:
                st.info(
                    "Odabrani model ne podrzava prikaz vjerojatnosti, "
                    "ali predikcija je uspjesno izracunata."
                )

except FileNotFoundError:
    st.error(
        "Nedostaje model ili rezultati treniranja. Prvo pokreni `python train.py`, "
        "pa zatim ponovno pokreni aplikaciju."
    )
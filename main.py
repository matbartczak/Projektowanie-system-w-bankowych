
import streamlit as st
import pandas as pd
import joblib

# Konfiguracja aplikacji
st.set_page_config(
    page_title="System oceny ryzyka kredytowego",
    page_icon="💳",
    layout="centered"
)

# Wczytanie modelu
@st.cache_resource
def load_model():
    return joblib.load("loan_pipeline.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(f"Nie udało się załadować modelu: {e}")
    st.stop()


# Interfejs
st.title("💳 System oceny ryzyka kredytowego")
st.write(
    "Wprowadź dane klienta, aby oszacować ryzyko niespłacenia kredytu."
)

st.divider()

with st.form("loan_form"):

    st.subheader("Dane klienta")

    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input(
            "Wiek klienta",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

        person_income = st.number_input(
            "Roczny dochód",
            min_value=0,
            value=60000,
            step=1000
        )

        home_ownership_options = {
            "Wynajem": "RENT",
            "Własność": "OWN",
            "Kredyt hipoteczny": "MORTGAGE",
            "Inne": "OTHER"
        }

        person_home_ownership_label = st.selectbox(
            "Status mieszkaniowy",
            options=list(home_ownership_options.keys())
        )

        person_home_ownership = home_ownership_options[person_home_ownership_label]

        person_emp_length = st.number_input(
            "Staż pracy (lata)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.5
        )

    loan_intent_options = {
        "Kredyt na cele osobiste": "PERSONAL",
        "Edukacja": "EDUCATION",
        "Cele medyczne": "MEDICAL",
        "Rozwój działalności gospodarczej": "VENTURE",
        "Remont lub ulepszenie domu": "HOMEIMPROVEMENT",
        "Konsolidacja zadłużenia": "DEBTCONSOLIDATION"
    }

    loan_intent_label = st.selectbox(
        "Cel kredytu",
        options=list(loan_intent_options.keys()),
        index=0
    )

    loan_intent = loan_intent_options[loan_intent_label]

    with col2:

        loan_amnt = st.number_input(
            "Kwota kredytu",
            min_value=1,
            value=10000,
            step=500
        )

        loan_int_rate = st.number_input(
            "Oprocentowanie (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1
        )

    submitted = st.form_submit_button(
        "Oceń ryzyko kredytowe",
        type="primary",
        use_container_width=True
    )


if submitted:

    # Dane wejściowe muszą mieć nazwy kolumn zgodne
    # z tymi, których użyto podczas trenowania pipeline'u.
    input_data = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_amnt/person_income,
        'loan_to_income_ratio' : loan_amnt/person_income,
        'loan_to_emp_length_ratio' :  person_emp_length/loan_amnt, 
        'int_rate_to_loan_amt_ratio' : loan_int_rate/loan_amnt
    }])

    input_data['age_group'] = pd.cut(input_data['person_age'],
                           bins=[20, 26, 36, 46, 56, 66],
                           labels=['20-25', '26-35', '36-45', '46-55', '56-65'])

    input_data['income_group'] = pd.cut(input_data['person_income'],
                              bins=[0, 25000, 50000, 75000, 100000, float('inf')],
                              labels=['low', 'low-middle', 'middle', 'high-middle', 'high'])

    input_data['loan_amount_group'] = pd.cut(input_data['loan_amnt'],
                                   bins=[0, 5000, 10000, 15000, float('inf')],
                                   labels=['small', 'medium', 'large', 'very large'])

    try:
        # Predykcja klasy
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Wynik oceny")

        if prediction == 1:
            st.error(
                "⚠️ Model wskazuje podwyższone ryzyko niespłacenia kredytu."
            )
        else:
            st.success(
                "✅ Model nie zaklasyfikował klienta jako zagrożonego "
                "niespłaceniem kredytu."
            )

        # Prawdopodobieństwo klasy, jeżeli model je udostępnia
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            classes = list(model.classes_)
            risk_class = 1

            if risk_class in classes:
                risk_probability = probabilities[
                    classes.index(risk_class)
                ]

                st.metric(
                    "Prawdopodobieństwo klasy ryzyka",
                    f"{risk_probability:.2%}"
                )

                st.progress(float(risk_probability))

                st.caption(
                    "Jest to prawdopodobieństwo oszacowane przez model, "
                    "a nie gwarancja rzeczywistego zachowania klienta."
                )

        with st.expander("Pokaż dane przekazane do modelu"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Błąd podczas predykcji: {e}")

"""Streamlit frontend for churn scoring.

Run it with:
    .venv/Scripts/python.exe -m streamlit run src/bank_churn/ui/app.py

The page reads the dataset summary from `data/processed/churn_processed.csv`, takes a
customer through a form, and asks the FastAPI service for a score. Until study 3 supplies
the model, the service answers 503 and the page says so.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import streamlit as st

from bank_churn.config.settings import DECISION_THRESHOLD, GENDERS, GEOGRAPHIES
from bank_churn.data.loader import churn_summary, processed_exists

DEFAULT_API_URL = os.environ.get("CHURN_API_URL", "http://127.0.0.1:8000")


def call_api(base_url: str, path: str, payload: dict | None = None, timeout: float = 10.0) -> tuple[int, dict]:
    """Small JSON client on the standard library, so the UI needs no extra dependency."""
    url = base_url.rstrip("/") + path
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"detail": body}
    except urllib.error.URLError as exc:
        return 0, {"detail": f"API not reachable at {url}: {exc.reason}"}


def render_sidebar() -> str:
    """Dataset facts on the left, API location and status at the bottom."""
    with st.sidebar:
        st.header("Dataset")
        if processed_exists():
            summary = churn_summary()
            st.metric("Customers", f"{summary['rows']:,}")
            st.metric("Churn rate", f"{summary['churn_rate']:.2%}")
            st.caption(f"{summary['churned']:,} customers left the bank")
        else:
            st.warning("Processed dataset missing. Run the notebooks first.")

        st.divider()
        st.header("API")
        base_url = st.text_input("Base URL", value=DEFAULT_API_URL)
        if st.button("Check API", use_container_width=True):
            code, body = call_api(base_url, "/health")
            if code == 200:
                st.success("API reachable")
                st.json(body)
            else:
                st.error(body.get("detail", "no response"))
        st.caption(f"Operating cut-off: {DECISION_THRESHOLD:.2f}")
    return base_url


def render_form() -> dict | None:
    """Collect one customer. Returns the payload when the user submits."""
    with st.form("customer"):
        left, right = st.columns(2)

        with left:
            credit_score = st.number_input("Credit score", 300, 850, 650)
            geography = st.selectbox("Country", GEOGRAPHIES)
            gender = st.selectbox("Gender", GENDERS)
            age = st.number_input("Age", 18, 100, 40)
            tenure = st.number_input("Tenure (years)", 0, 10, 5)

        with right:
            balance = st.number_input("Balance", 0.0, 300000.0, 75000.0, step=1000.0)
            num_of_products = st.number_input("Number of products", 1, 4, 1)
            has_cr_card = st.checkbox("Has a credit card", value=True)
            is_active_member = st.checkbox("Is an active member", value=True)
            estimated_salary = st.number_input(
                "Estimated salary", 0.0, 250000.0, 100000.0, step=1000.0
            )

        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if not submitted:
        return None

    return {
        "credit_score": int(credit_score),
        "geography": geography,
        "gender": gender,
        "age": int(age),
        "tenure": int(tenure),
        "balance": float(balance),
        "num_of_products": int(num_of_products),
        "has_cr_card": bool(has_cr_card),
        "is_active_member": bool(is_active_member),
        "estimated_salary": float(estimated_salary),
    }


def render_result(base_url: str, payload: dict) -> None:
    """Send the customer to the API and show whatever comes back."""
    code, body = call_api(base_url, "/predict", payload)

    if code == 200:
        probability = body["churn_probability"]
        left, right = st.columns(2)
        left.metric("Churn probability", f"{probability:.1%}")
        right.metric("Decision at cut-off", "at risk" if body["churn_flag"] else "likely to stay")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            f"cut-off {body['threshold']:.2f} | model {body['model_version']}"
        )
        return

    if code == 503:
        st.info(
            body.get("detail", "Model not trained yet.")
            + " The API contract is ready: the score appears once study 3 "
            "saves the fitted model."
        )
        return

    st.error(body.get("detail", f"request failed with status {code}"))


def main() -> None:
    st.set_page_config(page_title="Bank Churn Risk", page_icon="🏦", layout="wide")
    st.title("Bank customer churn risk")
    st.caption(
        "Enter one customer and get the probability that they leave. "
        "Built on the tuned XGBoost model from this project."
    )

    base_url = render_sidebar()

    tab_form, tab_about = st.tabs(["Score a customer", "About the model"])
    with tab_form:
        payload = render_form()
        if payload is not None:
            render_result(base_url, payload)
        else:
            st.caption("Fill the form and press Predict.")

    with tab_about:
        st.subheader("How the score is produced")
        st.markdown(
            """
- **Model:** XGBoost, tuned with Optuna on an F2 objective (recall weighted twice precision).
- **Features:** the 14 columns in `FEATURE_ORDER`, including the engineered `ZeroBalance`.
- **Cut-off:** chosen for the retention budget, not the statistical optimum; see `FINDINGS.md`.
- **Status:** the fitted model is pending study 3, so the API answers 503 for now.
            """
        )
        st.subheader("Strongest churn signals")
        st.markdown(
            """
1. Number of products (3-4 products churn far more)
2. Age (churners are older)
3. Active membership
4. Country (Germany churns about twice as often as France or Spain)
5. Zero balance (a *lower* risk segment)
            """
        )


if __name__ == "__main__":
    main()

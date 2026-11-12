"""Churn check page.

Run it with:
    PYTHONPATH=src .venv/Scripts/python.exe -m streamlit run src/bank_churn/ui/app.py

The page shows one customer form, checks the values with the same model the service uses, and
shows the result in plain words.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import streamlit as st
from pydantic import ValidationError

from bank_churn.api.schemas import CustomerFeatures
from bank_churn.config.settings import GENDERS, GEOGRAPHIES

DEFAULT_API_URL = os.environ.get("CHURN_API_URL", "http://127.0.0.1:8000")

FIELD_LABELS = {
    "credit_score": "Credit score",
    "geography": "Country",
    "gender": "Gender",
    "age": "Age",
    "tenure": "Years as a customer",
    "balance": "Account balance",
    "num_of_products": "Number of products",
    "has_cr_card": "Has a credit card",
    "is_active_member": "Active member",
    "estimated_salary": "Estimated salary",
}


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
        return 0, {"detail": f"no connection to {url}: {exc.reason}"}


def describe_error(error: dict) -> str:
    """Turn one pydantic error into a sentence that a user can act on."""
    kind = error.get("type", "")
    ctx = error.get("ctx") or {}
    if kind == "greater_than_equal":
        return f"must be {ctx.get('ge')} or more."
    if kind == "greater_than":
        return f"must be more than {ctx.get('gt')}."
    if kind == "less_than_equal":
        return f"must be {ctx.get('le')} or less."
    if kind == "less_than":
        return f"must be less than {ctx.get('lt')}."
    if kind in {"int_type", "int_parsing", "int_from_float"}:
        return "must be a whole number."
    if kind in {"float_type", "float_parsing"}:
        return "must be a number."
    if kind in {"enum", "literal_error"}:
        return "must be one of the values that the list shows."
    if kind == "missing":
        return "is necessary."
    return str(error.get("msg", "is not correct."))


def validate_payload(payload: dict) -> tuple[dict | None, list[str]]:
    """Check the values with the request model of the service. Returns the clean body and the problems."""
    try:
        customer = CustomerFeatures(**payload)
    except ValidationError as exc:
        problems = [
            f"**{FIELD_LABELS.get(str(error['loc'][0]), str(error['loc'][0]))}** {describe_error(error)}"
            for error in exc.errors()
        ]
        return None, problems
    return customer.model_dump(mode="json"), []


def render_sidebar() -> str:
    """Address of the scoring service and a button that checks it."""
    with st.sidebar:
        st.subheader("Service connection", icon=":material/settings_ethernet:")
        base_url = st.text_input(
            "Service address",
            value=DEFAULT_API_URL,
            key="base_url",
            help="Leave this value as it is, unless your administrator gives you an other address.",
        )
        if st.button("Check API", icon=":material/wifi_find:", width="stretch", key="check_api"):
            code, body = call_api(base_url, "/health")
            if code == 200 and body.get("model_ready"):
                st.success("Connected and ready", icon=":material/check_circle:")
            elif code == 200:
                st.warning("Connected, but not ready to score", icon=":material/hourglass_empty:")
            elif code == 0:
                st.error("No answer from the service", icon=":material/cloud_off:")
            else:
                st.error(f"The service answered with status {code}", icon=":material/error:")
        st.caption("Start the service before you use this page.")
    return base_url


def render_form() -> dict | None:
    """Collect one customer. Returns the raw values when the user submits."""
    with st.form("customer"):
        with st.container(border=True):
            st.subheader("Customer", icon=":material/person:")
            left, right = st.columns(2)

            with left:
                credit_score = st.number_input("Credit score", value=650, step=1)
                age = st.number_input("Age", value=40, step=1)
                tenure = st.number_input("Years as a customer", value=5, step=1)

            with right:
                geography = st.segmented_control(
                    "Country", GEOGRAPHIES, default=GEOGRAPHIES[0], key="country",
                )
                gender = st.segmented_control(
                    "Gender", GENDERS, default=GENDERS[0], key="gender",
                )

        with st.container(border=True):
            st.subheader("Account", icon=":material/account_balance:")
            left, right = st.columns(2)

            with left:
                balance = st.number_input("Account balance", value=75000.0, step=1000.0)
                num_of_products = st.number_input("Number of products", value=1, step=1)

            with right:
                estimated_salary = st.number_input("Estimated salary", value=100000.0, step=1000.0)
                has_cr_card = st.toggle("Has a credit card", value=True)
                is_active_member = st.toggle("Active member", value=True)

        submitted = st.form_submit_button(
            "Check churn risk",
            type="primary",
            width="stretch",
            icon=":material/person_search:",
            key="submit",
        )

    if not submitted:
        return None

    return {
        "credit_score": credit_score,
        "geography": geography,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "num_of_products": num_of_products,
        "has_cr_card": has_cr_card,
        "is_active_member": is_active_member,
        "estimated_salary": estimated_salary,
    }


def render_result(base_url: str, payload: dict) -> None:
    """Send the customer to the service and show the answer in plain words."""
    with st.spinner("Checking the customer"):
        code, body = call_api(base_url, "/predict", payload)

    if code == 200:
        probability = float(body["churn_probability"])
        at_risk = bool(body["churn_flag"])
        with st.container(border=True):
            st.subheader("Result", icon=":material/assignment_turned_in:")
            st.metric("Chance of leaving", f"{probability:.0%}")
            st.progress(probability)
            if at_risk:
                st.warning(
                    "Contact this customer. Offer a reason to stay.",
                    title="At risk",
                    icon=":material/warning:",
                )
            else:
                st.success(
                    "No action is necessary for this customer.",
                    title="Low risk",
                    icon=":material/check_circle:",
                )
        st.caption("The result is a guide. Speak to the customer before you act on it.")
        return

    if code == 503:
        st.info(
            "The service is not ready to score. Ask your administrator to start it, then try again.",
            title="Service not ready",
            icon=":material/hourglass_empty:",
        )
        return

    if code == 0:
        st.error(
            "The page cannot reach the service. Check the address in the sidebar and select Check API.",
            title="No connection",
            icon=":material/cloud_off:",
        )
        return

    st.error(
        f"The service did not accept the request. Status {code}.",
        title="Request failed",
        icon=":material/error:",
    )


def main() -> None:
    st.set_page_config(
        page_title="Customer churn check",
        page_icon=":material/person_search:",
        layout="centered",
    )
    st.title("Customer churn check", icon=":material/person_search:")
    st.caption(
        "Enter the details of one customer and select Check churn risk. "
        "The page shows the chance that this customer leaves the bank."
    )

    base_url = render_sidebar()

    payload = render_form()
    if payload is None:
        st.caption("The result appears here.")
        return

    clean, problems = validate_payload(payload)
    if clean is None:
        st.error(
            "\n".join(f"- {problem}" for problem in problems),
            title="Check these details",
            icon=":material/edit_note:",
        )
        return

    render_result(base_url, clean)


if __name__ == "__main__":
    main()

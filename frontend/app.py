import requests
import streamlit as st

DEFAULT_API_URL = "https://autonomous-data-detective-api.onrender.com"

st.set_page_config(
    page_title="Autonomous Data Detective",
    page_icon="🔎",
    layout="wide",
)

# ---------- Session state ----------

if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None

if "dataset_info" not in st.session_state:
    st.session_state.dataset_info = None

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


# ---------- Sidebar ----------

with st.sidebar:
    st.title("🔎 Data Detective")
    st.caption("AI-powered CSV investigation")

    api_url = st.text_input(
        "FastAPI base URL",
        value=DEFAULT_API_URL,
    ).strip().rstrip("/")

    st.divider()

    if st.session_state.dataset_info:
        st.success("Dataset uploaded")
        st.write(
            "**File:**",
            st.session_state.dataset_info["filename"],
        )
        st.caption(
            f"Dataset ID: {st.session_state.dataset_id}"
        )

        if st.button("Upload another dataset"):
            st.session_state.dataset_id = None
            st.session_state.dataset_info = None
            st.session_state.analysis_result = None
            st.rerun()
    else:
        st.info("Upload a CSV to begin.")

    st.divider()
    st.caption("Powered by FastAPI + LangGraph")


# ---------- Header ----------

st.title("Autonomous Data Detective")
st.write(
    "Upload a CSV dataset, ask a question in plain English, "
    "and let the AI investigation workflow analyze your data."
)

# ---------- API health ----------

try:
    health_response = requests.get(
        f"{api_url}/health",
        timeout=5,
    )
    health_response.raise_for_status()
    st.success("Backend connected")
except requests.RequestException:
    st.error(
        "Cannot connect to FastAPI. Start your backend and "
        "check the API URL in the sidebar."
    )

st.divider()


# ---------- Dataset upload ----------

st.subheader("1. Upload your dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a CSV file accepted by your FastAPI backend.",
)

if uploaded_file is not None:
    st.write(f"**Selected file:** {uploaded_file.name}")
    st.caption(
        f"File size: {uploaded_file.size:,} bytes"
    )

    if st.button("Upload CSV", type="primary"):
        try:
            with st.spinner("Validating and uploading CSV..."):
                response = requests.post(
                    f"{api_url}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "text/csv",
                        )
                    },
                    timeout=30,
                )

            if response.ok:
                upload_data = response.json()

                st.session_state.dataset_id = (
                    upload_data["dataset_id"]
                )
                st.session_state.dataset_info = upload_data
                st.session_state.analysis_result = None

                st.success(upload_data["message"])
                st.rerun()
            else:
                st.error(
                    f"Upload failed ({response.status_code}): "
                    f"{response.text}"
                )

        except requests.RequestException as exc:
            st.error(f"Upload request failed: {exc}")
        except (ValueError, KeyError) as exc:
            st.error(f"Unexpected upload response: {exc}")


# ---------- Uploaded dataset information ----------

if st.session_state.dataset_info:
    info = st.session_state.dataset_info

    st.subheader("Dataset information")

    col1, col2, col3 = st.columns(3)

    col1.metric("File", info["filename"])
    col2.metric("Size", f'{int(info["size_bytes"]):,} bytes')
    col3.metric("Status", "Uploaded")

    st.write("**Columns**")
    st.code(info["columns"] or "No columns returned")

    st.divider()

    # ---------- Ask a question ----------

    st.subheader("2. Ask your dataset")

    example_questions = [
        "Which product has the highest revenue?",
        "Compare average revenue by region.",
        "Are there any missing values?",
        "Give me a profile of the dataset.",
    ]

    selected_question = st.selectbox(
        "Example questions",
        ["Choose an example..."] + example_questions,
    )

    question = st.text_area(
        "Your question",
        value=(
            selected_question
            if selected_question != "Choose an example..."
            else ""
        ),
        placeholder=(
            "For example: Which product has the highest revenue?"
        ),
        height=100,
    )

    if st.button("Analyze dataset", type="primary"):
        if not question.strip():
            st.warning("Enter a question before running analysis.")
        else:
            try:
                with st.spinner(
                    "Investigating your dataset... This may take a while."
                ):
                    response = requests.post(
                        f"{api_url}/analyze",
                        json={
                            "dataset_id": st.session_state.dataset_id,
                            "question": question.strip(),
                        },
                        timeout=180,
                    )

                if response.ok:
                    result = response.json()

                    if not isinstance(result, dict):
                        st.error("The backend returned an unexpected response format.")
                    else:
                        st.session_state.analysis_result = result
                else:
                    st.error(
                        f"Analysis failed ({response.status_code}): "
                        f"{response.text}"
                    )

            except requests.Timeout:
                st.error(
                    "The request timed out. Check whether the "
                    "backend is still processing the analysis."
                )
            except requests.RequestException as exc:
                st.error(f"Analysis request failed: {exc}")
            except ValueError:
                st.error("The backend returned an invalid JSON response.")



# ---------- Analysis results ----------

import json


def parse_report(value):
    """Convert a JSON string to a Python object when possible."""
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    return {}


def display_items(title, items, icon_name=None):
    """Render a list of findings or recommendations."""
    if not items:
        return

    st.markdown(f"#### {title}")

    if isinstance(items, str):
        items = [items]

    for item in items:
        if isinstance(item, dict):
            st.write(item)
        else:
            st.markdown(f"- {item}")


result = st.session_state.analysis_result

if result:
    st.divider()
    st.header("3. Analysis results")

    # The final response may itself be a JSON-encoded report.
    report = parse_report(result.get("final_response"))

    # Some workflows may return the structured report separately.
    investigation = parse_report(
        result.get("investigation_result")
    )

    # Prefer the structured investigation report when available.
    if not report:
        report = investigation

    # If the workflow returned ordinary text, preserve it.
    final_response = result.get("final_response")
    if not report and isinstance(final_response, str):
        st.markdown(final_response)

    # ---------- Executive summary ----------

    summary = report.get("summary")

    if summary:
        st.subheader("Executive summary")
        st.info(summary)

    # ---------- Workflow metrics ----------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Workflow route",
        result.get("route", "Unknown"),
    )

    col2.metric(
        "Steps taken",
        result.get("steps_taken", 0),
    )

    claims = report.get("claims", [])
    validation_results = report.get("validation_results", [])

    verified_count = sum(
        1
        for validation in validation_results
        if validation.get("status") == "verified"
    )

    if validation_results:
        verification_display = (
            f"{verified_count}/{len(validation_results)}"
        )
        verification_label = "Verified claims"
    else:
        verification_display = "N/A"
        verification_label = "Claim verification"

    col3.metric(
        verification_label,
        verification_display,
    )

    # ---------- Findings ----------

    findings = report.get("findings", [])

    if findings:
        st.subheader("Key findings")
        for index, finding in enumerate(findings, start=1):
            st.markdown(f"**{index}.** {finding}")

    # ---------- Evidence ----------

    evidence = report.get("evidence", [])

    if evidence:
        with st.expander("Supporting evidence", expanded=True):
            for index, item in enumerate(evidence, start=1):
                st.markdown(f"- {item}")

    # ---------- Claim verification ----------

    if claims or validation_results:
        st.subheader("Claim verification")

        validation_by_id = {
            item.get("claim_id"): item
            for item in validation_results
            if isinstance(item, dict)
        }

        if claims:
            for claim in claims:
                if not isinstance(claim, dict):
                    st.write(claim)
                    continue

                claim_id = claim.get("claim_id")
                statement = claim.get("statement", "Claim statement unavailable")
                validation = validation_by_id.get(claim_id, {})
                status = validation.get("status", "unverifiable")

                with st.container(border=True):
                    st.markdown(f"**{statement}**")

                    if status == "verified":
                        st.success("Verified")
                    elif status == "invalid":
                        st.error("Invalid")
                    else:
                        st.warning("Unverifiable")

                    reason = validation.get("reason")
                    if reason:
                        st.caption(reason)

                    evidence_ids = validation.get("evidence_ids", [])
                    if evidence_ids:
                        st.caption(
                            "Evidence references: "
                            + ", ".join(evidence_ids)
                        )
        else:
            for validation in validation_results:
                status = validation.get("status", "unverifiable")
                st.write(
                    f"**{validation.get('claim_id', 'Claim')}** — "
                    f"{status}"
                )
                if validation.get("reason"):
                    st.caption(validation["reason"])

    # ---------- Limitations ----------

    display_items(
        "Limitations",
        report.get("limitations", []),
    )

    # ---------- Recommendations ----------

    display_items(
        "Recommendations",
        report.get("recommendations", []),
    )

    # ---------- Dataset profile ----------

    profile = result.get("profile_result")

    if profile:
        with st.expander("Dataset profile"):
            if isinstance(profile, (dict, list)):
                st.json(profile)
            else:
                st.write(profile)

    # ---------- Raw API response ----------

    with st.expander("Technical details — raw API response"):
        st.json(result)

st.divider()
st.caption(
    "Autonomous Data Detective | FastAPI + LangGraph + Streamlit"
)

"""
Page 6: Maintenance Recommendations & Human-in-the-Loop Review.
Displays prioritized maintenance protocols, linked physical evidence,
and human review interface (Approve / Reject / Request Further Analysis).
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone
from ..components.data_loader import get_sample_telemetry


def render_recommendations(dataset_type: str, selected_unit: str, api_url: str):
    st.subheader(f"🛠️ Prescriptive Maintenance Decisions & Review Workflow — {selected_unit}")

    st.warning(
        "**INDUSTRIAL ADVISORY & SAFETY GUARDRAIL:** The recommendations presented herein are decision-support "
        "advisories based on predictive models. They do NOT constitute autonomous equipment commands. "
        "Certified human maintenance engineering review and authorization are strictly required prior to physical equipment intervention."
    )

    # Fetch recent recommendations for this machine or generate on demand
    telemetry_data, _ = get_sample_telemetry(dataset_type, selected_unit)
    features_list = telemetry_data[-1, :].tolist() if telemetry_data.ndim >= 2 else telemetry_data.tolist()

    rec_data = None
    with st.spinner("Retrieving latest maintenance recommendation for this unit..."):
        try:
            # Query history for latest record for this unit
            r_hist = requests.get(f"{api_url}/api/history?machine_id={selected_unit}&limit=1", timeout=3.0)
            if r_hist.status_code == 200 and r_hist.json().get("records"):
                rec_data = r_hist.json()["records"][0]["recommendation_data"]
            else:
                # Trigger live recommendation via agentic workflow
                r_gen = requests.post(
                    f"{api_url}/api/agents/analyze",
                    json={"machine_id": selected_unit, "dataset_type": dataset_type, "sensor_data": features_list},
                    timeout=10.0
                )
                if r_gen.status_code == 200:
                    rec_data = r_gen.json().get("recommendation_output")
        except Exception as e:
            st.error(f"Failed to query recommendations: {str(e)}")

    if rec_data:
        rec_id = rec_data.get("recommendation_id", "REC-PENDING")
        priority = rec_data.get("priority", "Informational")
        risk_status = rec_data.get("risk_status", "nominal")
        human_req = rec_data.get("human_approval_required", True)

        # Priority card
        col_pri, col_stat = st.columns([2, 1])
        with col_pri:
            st.markdown(f"### Recommendation ID: `{rec_id}`")
            st.markdown(f"**Operational Priority Tier:** `{priority}` | **Risk State:** `{risk_status}`")
        with col_stat:
            # Current review status from DB
            review_status = "PENDING"
            reviewed_by = "N/A"
            try:
                r_chk = requests.get(f"{api_url}/api/history?machine_id={selected_unit}&limit=1", timeout=2.0)
                if r_chk.status_code == 200 and r_chk.json().get("records"):
                    review_status = r_chk.json()["records"][0].get("review_status", "PENDING")
                    reviewed_by = r_chk.json()["records"][0].get("reviewed_by") or "Unassigned"
            except Exception:
                pass

            if review_status == "APPROVED":
                st.success(f"STATUS: APPROVED\n\nBy: {reviewed_by}")
            elif review_status == "REJECTED":
                st.error(f"STATUS: REJECTED\n\nBy: {reviewed_by}")
            elif review_status == "FURTHER_ANALYSIS_REQUESTED":
                st.warning(f"STATUS: FURTHER ANALYSIS\n\nBy: {reviewed_by}")
            else:
                st.info("STATUS: PENDING REVIEW")

        st.markdown("---")
        st.markdown("#### 📋 Recommended Physical Action Items")
        actions = rec_data.get("recommended_actions", [])
        if actions:
            for idx, act in enumerate(actions):
                st.markdown(
                    f"""
                    <div style="background-color: #1E293B; border: 1px solid #334155; padding: 1rem; border-radius: 8px; margin-bottom: 0.75rem;">
                        <span style="font-weight: 700; color: #60A5FA;">Action {idx+1} ({act.get('action_id', 'ACT')}): {act.get('description', act.get('action_description', ''))}</span><br>
                        <span style="font-size: 0.85rem; color: #94A3B8;">Target Assembly: <strong>{act.get('target_subsystem', act.get('subsystem', 'N/A'))}</strong> | Urgency: <strong>{act.get('urgency', 'Planned')}</strong></span><br>
                        <span style="font-size: 0.80rem; color: #CBD5E1;">Standard Protocol: <em>{act.get('protocol_reference', 'Standard Maintenance Manual')}</em></span><br>
                        <span style="font-size: 0.80rem; color: #F59E0B;">Supporting Evidence: {act.get('justification', act.get('evidence_link', ''))}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No immediate physical actions required. Standard condition monitoring intervals maintained.")

        st.markdown("#### 🔬 Quantitative Evidence Basis")
        evidence = rec_data.get("supporting_evidence", [])
        if evidence:
            st.dataframe(pd.DataFrame(evidence), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### ✍️ Certified Engineer Authorization Panel")
        st.caption("Consequential maintenance interventions require explicit sign-off by certified reliability personnel.")

        with st.form(f"review_form_{rec_id}"):
            reviewer_name = st.text_input("Certified Engineer Name / Badge ID:", value=st.session_state.get("engineer_name", "Lead Propulsion Eng. Dave"))
            review_notes = st.text_area("Engineering Justification & Operational Notes:", value="Reviewed telemetry and verified borescope inspection schedule.")

            c_app, c_rej, c_fa = st.columns(3)
            with c_app:
                btn_approve = st.form_submit_button("✅ Approve Recommendation", type="primary", use_container_width=True)
            with c_rej:
                btn_reject = st.form_submit_button("❌ Reject Recommendation", use_container_width=True)
            with c_fa:
                btn_further = st.form_submit_button("🔍 Request Further Analysis", use_container_width=True)

            if btn_approve or btn_reject or btn_further:
                new_status = "APPROVED" if btn_approve else "REJECTED" if btn_reject else "FURTHER_ANALYSIS_REQUESTED"
                try:
                    r_rev = requests.post(
                        f"{api_url}/api/recommendations/review",
                        json={
                            "recommendation_id": rec_id,
                            "status": new_status,
                            "reviewed_by": reviewer_name,
                            "review_notes": review_notes
                        },
                        timeout=5.0
                    )
                    if r_rev.status_code == 200:
                        st.success(f"Recommendation `{rec_id}` updated to **{new_status}** by {reviewer_name}!")
                        st.session_state["engineer_name"] = reviewer_name
                        st.rerun()
                    else:
                        st.error(f"Review update failed: {r_rev.text}")
                except Exception as e:
                    st.error(f"Communication error: {str(e)}")
    else:
        st.info("Select a machine and run an analysis to view maintenance recommendations.")

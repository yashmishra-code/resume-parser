import time
import streamlit as st

from core import (
    extract_job_description,
    parse_resume,
    score_resume,
    generate_feedback,
    read_resume,
)

st.set_page_config(page_title="JobFit AI", page_icon="📄", layout="centered")
st.title("📄 JobFit AI")


def render_match(match):
    st.write(f"**Verdict:** {match.verdict}")

    met = match.experience_requirement_met
    met_label = "Yes" if met is True else "No" if met is False else "Unclear"
    st.write(f"**Experience requirement met:** {met_label}")

    if match.matching_skills:
        st.write("**Matching skills:**")
        st.markdown("\n".join(f"- {s}" for s in match.matching_skills))
    else:
        st.write("**Matching skills:** none found")

    if match.missing_skills:
        st.write("**Missing skills:**")
        st.markdown("\n".join(f"- {s}" for s in match.missing_skills))
    else:
        st.write("**Missing skills:** none")

MAX_RESUMES = 10
SLEEP_BETWEEN_CALLS = 5  # keep this to avoid rate-limit issues on Groq's free tier

mode = st.radio(
    "I am a...",
    ["HR reviewing candidates", "Individual checking my own resume"],
)

jd_text = st.text_area("Paste the job description here", height=200)

st.divider()

# ---------------- HR MODE ----------------
if mode == "HR reviewing candidates":
    files = st.file_uploader(
        f"Upload resumes (PDF or DOCX, max {MAX_RESUMES})",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if files and len(files) > MAX_RESUMES:
        st.warning(f"Please upload at most {MAX_RESUMES} resumes. You uploaded {len(files)}.")

    run = st.button("Rank candidates", type="primary")

    if run:
        if not jd_text.strip():
            st.error("Please paste a job description first.")
        elif not files:
            st.error("Please upload at least one resume.")
        elif len(files) > MAX_RESUMES:
            st.error(f"Too many resumes. Max is {MAX_RESUMES}.")
        else:
            with st.spinner("Reading job description..."):
                job = extract_job_description(jd_text)

            results = []
            progress = st.progress(0, text="Starting...")

            for i, f in enumerate(files):
                progress.progress(i / len(files), text=f"Processing {f.name}...")
                resume_text = read_resume(f)
                if not resume_text:
                    st.warning(f"Could not read {f.name}, skipping.")
                    continue

                parsed = parse_resume(resume_text)
                time.sleep(SLEEP_BETWEEN_CALLS)
                match = score_resume(job, parsed)
                time.sleep(SLEEP_BETWEEN_CALLS)

                results.append({
                    "name": match.candidate_name or parsed.name or f.name,
                    "score": match.score,
                    "match": match,
                })

            progress.progress(1.0, text="Done")
            results.sort(key=lambda c: c["score"], reverse=True)

            st.subheader("Ranked candidates")
            for c in results:
                with st.expander(f"{c['name']} — {c['score']}%"):
                    render_match(c["match"])

# ---------------- INDIVIDUAL MODE ----------------
else:
    file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

    run = st.button("Check my fit", type="primary")

    if run:
        if not jd_text.strip():
            st.error("Please paste a job description first.")
        elif not file:
            st.error("Please upload your resume.")
        else:
            with st.spinner("Reading job description..."):
                job = extract_job_description(jd_text)

            resume_text = read_resume(file)
            if not resume_text:
                st.error("Could not read that file.")
            else:
                with st.spinner("Parsing your resume..."):
                    resume = parse_resume(resume_text)
                time.sleep(SLEEP_BETWEEN_CALLS)

                with st.spinner("Scoring your match..."):
                    match = score_resume(job, resume)
                time.sleep(SLEEP_BETWEEN_CALLS)

                with st.spinner("Generating feedback..."):
                    feedback = generate_feedback(job, resume, match)

                st.metric("Match score", f"{match.score}%")
                st.subheader("Details")
                render_match(match)
                st.subheader("How to improve your resume for this role")
                st.markdown(feedback)
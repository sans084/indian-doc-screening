"""
app.py
-------
AI-Based Fake Identity & Document Screening System
Smart India Hackathon - Cybersecurity & Blockchain track

Two distinct use cases, chosen up front, because they don't need the same
steps:

  DOCUMENT-ONLY SCREENING
    "Is this specific document image fake/tampered?" - no live person
    needed. Upload -> OCR -> Forensic Screening -> Final Report.

  FULL IDENTITY VERIFICATION
    "Does the person in front of the camera match this document, and is
    the document itself genuine?" - the KYC/onboarding scenario, where
    a live selfie makes sense because the document holder IS the one
    using the app. Upload -> OCR -> Forensic Screening -> Live Selfie
    -> Face Match -> Final Report.

Both modes end by recording the outcome to the blockchain-style ledger.

All processing is local. Uploaded images are written to a per-session
temp directory and deleted at the end of the session - no document image
or selfie is ever persisted to disk long-term or sent anywhere. Only a
SHA-256 hash of the document (not the image itself) is written to the
verification ledger.

SCOPE NOTE (say this out loud in any demo): forensic screening here checks
whether ONE uploaded image is internally consistent - not whether the
document is a real UIDAI record. A well-fabricated document with a
checksum-valid-looking number and no visible splice can still score
moderately. See forgery_detection.py's module docstring for the full
explanation of what each check can and can't tell you.
"""

import os
import tempfile

import streamlit as st

from ocr_extraction import extract_fields, detect_document_type
from forgery_detection import DocumentForgeryDetector
from face_match import match_faces
from blockchain_ledger import VerificationLedger


st.set_page_config(page_title="Document Authenticity & Identity Screening", layout="wide")

# ---------------------------------------------------------------------------
# Design system - "forensic console": dark ink background, IBM Plex type,
# mono for data/hashes, a left-hand step rail instead of a card-grid layout.
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #0B1220;
    --panel: #131C2E;
    --panel-line: #24304A;
    --text: #E6E9EF;
    --muted: #8892A6;
    --accent: #5B8DEF;
    --pass: #3DDC97;
    --flag: #FF6B5D;
    --review: #F4B942;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: var(--ink); color: var(--text); }

section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--panel-line); }

.brand { padding: 6px 0 22px 0; border-bottom: 1px solid var(--panel-line); margin-bottom: 18px; }
.brand .tag { font-family:'IBM Plex Mono', monospace; font-size: 11px; color: var(--accent); letter-spacing: 0.04em; }
.brand h2 { font-size: 19px; margin: 4px 0 0 0; color: var(--text); }

.step-item { padding: 9px 10px; border-radius: 4px; margin-bottom: 4px; font-size: 14px; }
.step-item.done { color: var(--pass); }
.step-item.active { background: rgba(91,141,239,0.14); color: var(--accent); font-weight: 600; }
.step-item.pending { color: var(--muted); }

.page-header { border-bottom: 1px solid var(--panel-line); padding-bottom: 16px; margin-bottom: 24px; }
.page-header .eyebrow { font-family:'IBM Plex Mono', monospace; font-size: 12px; color: var(--muted); letter-spacing: 0.03em; }
.page-header h1 { font-size: 28px; margin: 4px 0 6px 0; }
.page-header p { color: var(--muted); font-size: 14.5px; max-width: 640px; margin: 0; }

.field-row { display:flex; justify-content:space-between; padding: 8px 0; border-bottom: 1px dashed var(--panel-line); font-size: 14px; }
.field-row .k { color: var(--muted); }
.field-row .v { font-family:'IBM Plex Mono', monospace; color: var(--text); }

.ledger-block {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    background: var(--panel);
    border: 1px solid var(--panel-line);
    border-radius: 4px;
    padding: 12px 14px;
    margin-bottom: 8px;
    color: var(--muted);
}
.ledger-block .hash { color: #9FE8C7; word-break: break-all; }

.verdict-banner {
    border-radius: 6px;
    padding: 18px 22px;
    margin-bottom: 20px;
    border: 1px solid var(--panel-line);
}
.verdict-banner.genuine { border-left: 4px solid var(--pass); }
.verdict-banner.review { border-left: 4px solid var(--review); }
.verdict-banner.fake { border-left: 4px solid var(--flag); }
.verdict-banner h2 { margin: 0 0 4px 0; font-size: 22px; }
.verdict-banner p { margin: 0; color: var(--muted); font-size: 14px; }

.mode-card {
    background: var(--panel);
    border: 1px solid var(--panel-line);
    border-radius: 6px;
    padding: 22px 24px;
    margin-bottom: 14px;
}
.mode-card h3 { margin: 0 0 6px 0; font-size: 17px; }
.mode-card p { color: var(--muted); font-size: 13.5px; margin: 0 0 4px 0; }

.stButton>button {
    background: var(--accent); color: #0B1220; border: none; font-weight: 600;
    border-radius: 4px; padding: 8px 18px;
}
.stButton>button:hover { background: #7BA3F5; color: #0B1220; }
</style>
"""

RESET_KEYS = ("doc_path", "selfie_path", "ocr_result", "doc_type", "forgery_report", "face_result", "ledger_block")


def init_session():
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "mode" not in st.session_state:
        st.session_state.mode = None  # "document_only" | "full"
    if "workdir" not in st.session_state:
        st.session_state.workdir = tempfile.mkdtemp(prefix="docscreen_")
    if "ledger" not in st.session_state:
        st.session_state.ledger = VerificationLedger(
            os.path.join(st.session_state.workdir, "..", "verification_ledger.json")
        )
    for key in RESET_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None


def steps_for_mode():
    """The ordered list of step functions for the current mode. Index in
    this list IS st.session_state.step - keep the two in sync."""
    if st.session_state.mode == "document_only":
        return [step_mode, step_upload, step_ocr, step_forgery, step_report]
    elif st.session_state.mode == "full":
        return [step_mode, step_upload, step_ocr, step_forgery, step_selfie, step_face_match, step_report]
    else:
        return [step_mode]


def sidebar_labels():
    if st.session_state.mode == "document_only":
        return ["Choose Mode", "Upload Document", "Text Extraction", "Forensic Screening", "Final Report"]
    elif st.session_state.mode == "full":
        return ["Choose Mode", "Upload Document", "Text Extraction", "Forensic Screening", "Live Selfie", "Face Match", "Final Report"]
    else:
        return ["Choose Mode"]


def render_sidebar():
    st.sidebar.markdown(
        """
        <div class="brand">
            <div class="tag">SIH · CYBERSECURITY &amp; BLOCKCHAIN</div>
            <h2>Identity &amp; Document<br/>Screening Console</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for i, label in enumerate(sidebar_labels()):
        if i < st.session_state.step:
            cls, marker = "done", "✓"
        elif i == st.session_state.step:
            cls, marker = "active", "›"
        else:
            cls, marker = "pending", "·"
        st.sidebar.markdown(f'<div class="step-item {cls}">{marker}  {label}</div>', unsafe_allow_html=True)

    st.sidebar.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    valid, broken_at = st.session_state.ledger.is_chain_valid()
    status = "Chain intact" if valid else f"⚠ Broken at block {broken_at}"
    st.sidebar.markdown(
        f"<div class='step-item pending' style='font-family:IBM Plex Mono, monospace; font-size:11.5px;'>"
        f"LEDGER: {status}<br/>{st.session_state.ledger.block_count()} records</div>",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str):
    total = len(sidebar_labels())
    eyebrow = f"STEP {st.session_state.step + 1} / {total}"
    st.markdown(
        f"""
        <div class="page-header">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_upload(uploaded_file, dest_name: str) -> str:
    ext = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    path = os.path.join(st.session_state.workdir, dest_name + ext)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def advance():
    st.session_state.step += 1
    st.rerun()


# ---------------------------------------------------------------------------
# Step screens
# ---------------------------------------------------------------------------

def step_mode():
    page_header("What do you want to do?",
                "These need different steps, so pick the one that matches your situation.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="mode-card">
                <h3>📄 Screen a document</h3>
                <p>You have a document image and want to check if it's fake or tampered —
                no live person needed. E.g. reviewing a submitted ID as an institution.</p>
                <p>Steps: Upload → OCR → Forensic Screening → Report</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Screen a document only →", key="mode_doc"):
            st.session_state.mode = "document_only"
            advance()
    with col2:
        st.markdown(
            """
            <div class="mode-card">
                <h3>🪪 Full identity verification</h3>
                <p>You (the document holder) are here to prove your own identity —
                the app will also compare your live selfie against the document photo.
                E.g. onboarding / KYC.</p>
                <p>Steps: Upload → OCR → Forensic Screening → Selfie → Face Match → Report</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Full identity verification →", key="mode_full"):
            st.session_state.mode = "full"
            advance()


def step_upload():
    page_header("Upload identity document",
                "Aadhaar, PAN, Driving Licence, or Passport. Use a clear, well-lit photo — avoid screenshots where possible.")
    uploaded = st.file_uploader("Document image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded:
        st.image(uploaded, width=360)
        if st.button("Continue to text extraction →"):
            st.session_state.doc_path = save_upload(uploaded, "document")
            advance()


def step_ocr():
    page_header("Text extraction",
                "OCR reads the printed fields off the document so they can be cross-checked in the next step.")
    with st.spinner("Running OCR..."):
        result = extract_fields(st.session_state.doc_path)
        doc_type = detect_document_type(result.raw_text)
    st.session_state.ocr_result = result
    st.session_state.doc_type = doc_type

    if result.is_blurry:
        st.warning(f"Image sharpness is low (score {result.sharpness_score:.0f}). Extraction may be unreliable — consider re-uploading a clearer photo.")

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(st.session_state.doc_path, width=320)
    with col2:
        fields = [
            ("Document type", doc_type.replace("_", " ").title()),
            ("Name", result.name or "—"),
            ("Date of birth", result.dob or "—"),
            ("Age", str(result.age) if result.age else "—"),
            ("Gender", result.gender or "—"),
            ("ID number", result.id_number or "—"),
        ]
        for k, v in fields:
            st.markdown(f'<div class="field-row"><span class="k">{k}</span><span class="v">{v}</span></div>', unsafe_allow_html=True)

    if st.button("Continue to forensic screening →"):
        advance()


def step_forgery():
    page_header("Forensic authenticity screening",
                "QR cross-verification, Aadhaar checksum validation, compression-error analysis, duplicated-region detection, and metadata forensics.")

    with st.spinner("Running forensic checks..."):
        detector = DocumentForgeryDetector()
        report = detector.analyze(st.session_state.doc_path, ocr_text=st.session_state.ocr_result.raw_text)
    st.session_state.forgery_report = report

    verdict_class = {"Likely Genuine": "genuine", "Needs Manual Review": "review", "Likely Fake / Tampered": "fake"}.get(report.verdict, "review")
    st.markdown(
        f"""
        <div class="verdict-banner {verdict_class}">
            <h2>{report.verdict} — {report.authenticity_score:.0f}/100</h2>
            <p>Composite score from {len(report.checks)} independent forensic checks, weighted by reliability.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for check in report.checks:
        tag = "✓ CLEAR" if check.passed else "⚠ FLAGGED"
        color = "var(--pass)" if check.passed else "var(--flag)"
        st.markdown(
            f"""
            <div class="ledger-block">
                <span style="color:{color}; font-weight:600;">{tag}</span> &nbsp; <b style="color:var(--text)">{check.name}</b>
                &nbsp;<span style="color:var(--muted)">(score {check.score:.0f}/100, weight {check.weight:.0%})</span><br/>
                {check.details}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if report.authenticity_score < 40:
        st.error("Document failed forensic screening. Proceeding is not recommended — flag for manual review.")

    # Duplicate-submission check - this is the blockchain ledger earning its
    # keep beyond just tamper-evidence: if this EXACT document image has
    # been verified before, that's a real fraud signal (a stolen/leaked ID
    # photo being reused across multiple verification attempts).
    prior_records = st.session_state.ledger.lookup_document(st.session_state.doc_path)
    st.session_state.prior_verifications = prior_records
    if prior_records:
        import datetime
        lines = []
        for rec in prior_records:
            ts = datetime.datetime.fromtimestamp(rec["timestamp"]).strftime("%d %b %Y, %H:%M")
            lines.append(f"• Block #{rec['index']} — {ts} — verdict: {rec['data']['authenticity_verdict']}")
        st.warning(
            f"⚠ **This exact document has already been verified {len(prior_records)} time(s) before**, "
            f"per the ledger:\n\n" + "\n".join(lines) +
            "\n\nA legitimate document shouldn't be verified repeatedly through separate applications. "
            "This warrants manual review before proceeding."
        )

    st.caption("Note: these checks confirm this image is internally consistent (not visibly edited) — "
               "they don't confirm the document is a real UIDAI/government record.")

    next_label = "Continue to final report →" if st.session_state.mode == "document_only" else "Continue to live selfie →"
    if st.button(next_label):
        advance()


def step_selfie():
    page_header("Live selfie capture",
                "Used only to compare against the face on the document. Deleted at the end of this session.")
    photo = st.camera_input("Take a selfie", label_visibility="collapsed")
    if photo:
        if st.button("Continue to face match →"):
            st.session_state.selfie_path = save_upload(photo, "selfie")
            advance()


def step_face_match():
    page_header("Face match",
                "Comparing the face on the document against the live selfie.")
    with st.spinner("Matching faces..."):
        result = match_faces(st.session_state.doc_path, st.session_state.selfie_path, sensitivity="balanced")
    st.session_state.face_result = result

    if result.error:
        st.warning(result.error)
    else:
        verdict_class = "genuine" if result.matched else "fake"
        label = "Face Match" if result.matched else "Face Mismatch"
        st.markdown(
            f"""
            <div class="verdict-banner {verdict_class}">
                <h2>{label} — {result.confidence:.0f}% confidence</h2>
                <p>Distance {result.distance:.3f} against threshold {result.threshold:.3f}.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("Continue to final report →"):
        advance()


def step_report():
    page_header("Final report & blockchain record",
                "Combined outcome, recorded to the tamper-evident verification ledger.")

    report = st.session_state.forgery_report
    face = st.session_state.face_result  # None in document_only mode

    if face is not None and not face.error:
        combined_ok = report.authenticity_score >= 50 and face.matched
        face_line = f"Face match {'passed' if face.matched else 'failed'} ({face.confidence:.0f}% confidence)"
    elif face is not None and face.error:
        combined_ok = False
        face_line = f"Face match could not be completed: {face.error}"
    else:
        combined_ok = report.authenticity_score >= 50
        face_line = "Face match not applicable — document-only screening"

    prior_count = len(st.session_state.get("prior_verifications") or [])
    if prior_count > 0:
        combined_ok = False  # duplicate submission overrides an otherwise-clean result

    overall_class = "genuine" if combined_ok else "fake"
    overall_label = "VERIFIED" if combined_ok else "NOT VERIFIED"

    st.markdown(
        f"""
        <div class="verdict-banner {overall_class}">
            <h2>{overall_label}</h2>
            <p>Authenticity score {report.authenticity_score:.0f}/100 · {face_line}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if prior_count > 0:
        st.warning(f"⚠ This document was already verified {prior_count} time(s) before this session — see the ledger block below for prior records.")

    if st.session_state.ledger_block is None:
        block = st.session_state.ledger.add_verification_record(
            document_path=st.session_state.doc_path,
            document_type=st.session_state.doc_type,
            authenticity_score=report.authenticity_score,
            authenticity_verdict=report.verdict,
            face_match_score=(face.confidence if (face is not None and not face.error) else None),
            face_match_passed=(face.matched if (face is not None and not face.error) else None),
        )
        st.session_state.ledger_block = block

    block = st.session_state.ledger_block
    st.markdown(
        f"""
        <div class="ledger-block">
            <b style="color:var(--text)">Ledger record — block #{block.index}</b><br/>
            document hash: <span class="hash">{block.data['document_hash'][:48]}...</span><br/>
            block hash: <span class="hash">{block.hash[:48]}...</span><br/>
            previous hash: <span class="hash">{block.previous_hash[:48]}...</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    valid, broken_at = st.session_state.ledger.is_chain_valid()
    if valid:
        st.success(f"Ledger integrity verified — {st.session_state.ledger.block_count()} records, chain intact.")
    else:
        st.error(f"Ledger integrity check FAILED at block {broken_at}. Records may have been tampered with.")

    if st.button("Start new verification"):
        for key in RESET_KEYS:
            st.session_state[key] = None
        st.session_state.mode = None
        st.session_state.step = 0
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    init_session()
    render_sidebar()

    steps = steps_for_mode()
    # Guard against a stale step index (e.g. after switching mode) pointing
    # past the end of the current mode's step list.
    if st.session_state.step >= len(steps):
        st.session_state.step = len(steps) - 1
    steps[st.session_state.step]()


if __name__ == "__main__":
    main()
# Identity & Document Screening Console

**AI-Based Fake Identity & Document Screening System**
Smart India Hackathon — Cybersecurity & Blockchain track

An end-to-end pipeline that screens Indian identity documents (Aadhaar,
PAN, Driving Licence, Passport) for forgery, matches the document photo
against a live selfie, and records every verification outcome on a
tamper-evident hash-chain ledger.

## Why this exists

Most demo-stage "ID verification" projects stop at face matching. This
one is built around the actual problem statement: **screening for fake
documents**, not just confirming a face looks similar. It adds a real
forensic layer and a real (if lightweight) blockchain-style audit trail,
which is what the theme calls for.

## Pipeline

```
Upload document
      │
      ▼
OCR text extraction  (Tesseract)
      │
      ▼
Forensic authenticity screening
  ├─ QR cross-verification (secure QR data vs printed OCR text)
  ├─ Error Level Analysis (JPEG recompression artifact detection)
  ├─ Copy-move detection (duplicated-region / displacement clustering)
  └─ Metadata forensics (EXIF, editor fingerprints, screenshot signals)
      │
      ▼
Live selfie capture
      │
      ▼
Face match  (DeepFace, quality-gated)
      │
      ▼
Combined verdict → recorded on the verification ledger (SHA-256 hash chain)
      │
      ▼
Final report + ledger integrity check
```

## Modules

| File                    | Responsibility                                                        |
|-------------------------|-------------------------------------------------------------------------|
| `app.py`                | Streamlit UI — step-wizard console tying the whole pipeline together    |
| `ocr_extraction.py`     | Tesseract-based field extraction (name, DOB, gender, ID number) + blur detection |
| `forgery_detection.py`  | Forensic authenticity engine — QR check, ELA, copy-move, metadata       |
| `face_match.py`         | DeepFace wrapper with quality gating and adjustable sensitivity         |
| `blockchain_ledger.py`  | Local tamper-evident hash-chain ledger for verification records          |

Each module works standalone (run any of them directly, e.g.
`python forgery_detection.py your_image.jpg`) and has no hidden coupling
to the Streamlit layer — useful if you later want to expose this as a
FastAPI backend instead.

## Setup

```bash
git clone <your-repo-url>
cd indian-doc-screening
pip install -r requirements.txt
```

Install Tesseract OCR and libzbar (see comments in `requirements.txt`):

```bash
# Linux
sudo apt-get install tesseract-ocr libzbar0

# macOS
brew install tesseract zbar

# Windows
# Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# libzbar ships inside the pyzbar wheel, no extra step
```

Run it:

```bash
streamlit run app.py
```

## Why a hash-chain ledger and not a "real" blockchain?

For a hackathon demo, the property that matters is **tamper-evidence**:
once a verification result is written, altering it after the fact is
detectable. `blockchain_ledger.py` implements exactly that — each block
stores a hash of its own contents plus the previous block's hash, so
changing any past record breaks the chain (this is demonstrated in the
module's own self-test: `python blockchain_ledger.py`).

Only a SHA-256 **hash** of the document image is ever stored in the
ledger — never the image itself, and never extracted PII — so the ledger
carries no personal data even though it proves a specific document was
screened and what the outcome was.

This is deliberately swappable: `add_verification_record` /
`is_chain_valid` are the only two calls `app.py` makes into the ledger.
Replacing the local JSON-backed chain with a Hyperledger Fabric channel or
an Ethereum smart contract (via `web3.py` against Ganache or a testnet)
later only means reimplementing those two functions — the rest of the
app doesn't need to change.

## Privacy

- All processing happens locally — no image, selfie, or extracted text
  ever leaves the machine running the app.
- Uploaded images are written to a per-session temp directory and are not
  persisted after the session ends.
- The verification ledger stores hashes and outcome metadata only.

## Roadmap

- [ ] Swap the local hash-chain for a real smart-contract-backed ledger (Ganache/testnet)
- [ ] REST API (FastAPI) alongside the Streamlit UI
- [ ] Multi-document batch verification + institutional admin dashboard
- [ ] Face liveness detection (blink/head-turn) to prevent photo-of-a-photo spoofing
- [ ] Template/layout matching against known genuine document layouts

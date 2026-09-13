"""
blockchain_ledger.py
---------------------
A lightweight, dependency-free hash-chain ledger for recording document
verification outcomes in a tamper-evident way.

This is deliberately NOT a full distributed blockchain (no consensus, no
network of nodes) - for a hackathon demo, a locally-verifiable hash chain
demonstrates the core property that matters for this problem statement:
once a verification result is recorded, it cannot be silently altered
without breaking the chain, and that's checkable by anyone with the
ledger file.

Each block stores:
  - index            position in the chain
  - timestamp        when the record was added
  - data             the verification result (never raw PII/images -
                      only a hash of the document + the outcome)
  - previous_hash     hash of the prior block
  - hash              hash of this block's own contents

Swapping this out for a real network (Hyperledger Fabric, or an Ethereum
smart contract via web3.py against Ganache/a testnet) later only requires
replacing `add_block` / `is_chain_valid` with contract calls - the calling
code in app.py doesn't need to change.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


LEDGER_PATH_DEFAULT = "verification_ledger.json"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str) -> str:
    """Content hash of a document image - used so we record proof a
    specific document was verified without ever storing the document
    itself in the ledger."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Block:
    index: int
    timestamp: float
    data: dict
    previous_hash: str
    hash: str = ""

    def compute_hash(self) -> str:
        payload = {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
        }
        encoded = json.dumps(payload, sort_keys=True).encode()
        return sha256_hex(encoded)


class VerificationLedger:
    """
    File-backed hash chain. Safe for single-process demo use; for a real
    deployment this file would be replaced by a proper blockchain client.
    """

    def __init__(self, ledger_path: str = LEDGER_PATH_DEFAULT):
        self.ledger_path = ledger_path
        self.chain: list[Block] = []
        self._load_or_init()

    # -- persistence -----------------------------------------------------

    def _load_or_init(self):
        if os.path.exists(self.ledger_path):
            with open(self.ledger_path, "r") as f:
                raw = json.load(f)
            self.chain = [Block(**b) for b in raw]
        else:
            genesis = Block(index=0, timestamp=time.time(),
                             data={"type": "genesis", "note": "Verification ledger initialized"},
                             previous_hash="0")
            genesis.hash = genesis.compute_hash()
            self.chain = [genesis]
            self._save()

    def _save(self):
        with open(self.ledger_path, "w") as f:
            json.dump([asdict(b) for b in self.chain], f, indent=2)

    # -- core API ----------------------------------------------------------

    def add_verification_record(
        self,
        document_path: str,
        document_type: str,
        authenticity_score: float,
        authenticity_verdict: str,
        face_match_score: Optional[float] = None,
        face_match_passed: Optional[bool] = None,
        extra: Optional[dict] = None,
    ) -> Block:
        """
        Records a verification outcome on the chain. Only a hash of the
        document is stored - never the image or extracted PII - so the
        ledger itself carries no personal data, only tamper-evident proof
        that a specific document (identified by its hash) was screened and
        what the result was.
        """
        record = {
            "type": "verification",
            "document_hash": hash_file(document_path),
            "document_type": document_type,
            "authenticity_score": round(authenticity_score, 1),
            "authenticity_verdict": authenticity_verdict,
            "face_match_score": round(face_match_score, 1) if face_match_score is not None else None,
            "face_match_passed": face_match_passed,
        }
        if extra:
            record["extra"] = extra

        prev = self.chain[-1]
        block = Block(
            index=prev.index + 1,
            timestamp=time.time(),
            data=record,
            previous_hash=prev.hash,
        )
        block.hash = block.compute_hash()
        self.chain.append(block)
        self._save()
        return block

    # -- integrity check -----------------------------------------------------

    def is_chain_valid(self) -> tuple[bool, Optional[int]]:
        """
        Walks the chain and confirms every block's stored hash matches a
        fresh recomputation, and that previous_hash links are intact.
        Returns (valid, first_broken_index_or_None).
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prior = self.chain[i - 1]

            if current.hash != current.compute_hash():
                return False, i  # block contents were altered after the fact
            if current.previous_hash != prior.hash:
                return False, i  # chain link broken - a block was removed/reordered

        return True, None

    def lookup_document(self, document_path: str) -> list[dict]:
        """Find all prior verification records for a given document hash -
        useful to detect re-submission of the same document under a
        different claimed identity."""
        target_hash = hash_file(document_path)
        return [
            asdict(b) for b in self.chain
            if b.data.get("type") == "verification" and b.data.get("document_hash") == target_hash
        ]

    def recent_records(self, limit: int = 20) -> list[dict]:
        records = [asdict(b) for b in self.chain if b.data.get("type") == "verification"]
        return list(reversed(records))[:limit]

    def block_count(self) -> int:
        return len(self.chain) - 1  # exclude genesis


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ledger_path = os.path.join(tmp, "test_ledger.json")
        doc_path = os.path.join(tmp, "doc.txt")
        with open(doc_path, "w") as f:
            f.write("fake document bytes")

        ledger = VerificationLedger(ledger_path)
        b1 = ledger.add_verification_record(doc_path, "aadhaar", 86.1, "Likely Genuine", 92.3, True)
        b2 = ledger.add_verification_record(doc_path, "aadhaar", 57.0, "Needs Manual Review", 40.1, False)

        print(f"Chain length: {ledger.block_count()} verification blocks")
        valid, broken_at = ledger.is_chain_valid()
        print(f"Chain valid: {valid}")

        # tamper test - alter a stored record after the fact
        ledger.chain[1].data["authenticity_score"] = 99.9
        valid, broken_at = ledger.is_chain_valid()
        print(f"After tampering block 1: valid={valid}, broken_at={broken_at}")

        print(f"Records for this document: {len(ledger.lookup_document(doc_path))}")

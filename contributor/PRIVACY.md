# Contributor privacy contract

The developer extension is local-only by default. It does not send images, page URLs, scores, or evidence to a server. A contributor can download a prepared JSON file and deliver it through an approved maintainer process.

If a future first-party quarantine service is approved, the intended lifecycle is: private quarantine → human review → raw image deletion after review → retain only a content hash, detector score, sanitized origin/path, contributor attestation, review decision, deletion record, and training lineage. Access must be restricted and auditable.

The extension strips URL credentials, query strings, and fragments. “Anonymous” is not promised: a future service may retain security and abuse-prevention logs. No public GitHub or Hugging Face raw upload is supported.

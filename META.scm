;; SPDX-License-Identifier: MIT OR AGPL-3.0-or-later
;; SPDX-FileCopyrightText: 2024-2025 hyperpolymath
;;; META.scm — zerostep

(define-module (zerostep meta)
  #:export (architecture-decisions development-practices design-rationale))

(define architecture-decisions
  '((adr-001
     (title . "RSR Compliance")
     (status . "accepted")
     (date . "2025-12-15")
     (context . "Project needs consistent, secure, reproducible infrastructure")
     (decision . "Follow Rhodium Standard Repository guidelines")
     (consequences . ("RSR Gold target" "SHA-pinned actions" "SPDX headers" "Multi-platform CI")))
    (adr-002
     (title . "Cryptographic Integrity")
     (status . "accepted")
     (date . "2025-12-15")
     (context . "Dataset integrity verification needed for ML training")
     (decision . "Use SHAKE256 (FIPS 202) for all file checksums")
     (consequences . ("FIPS compliant" "Extensible output" "No custom crypto")))
    (adr-003
     (title . "Formal Verification")
     (status . "accepted")
     (date . "2025-12-15")
     (context . "Split algorithms must guarantee disjoint, complete partitions")
     (decision . "Isabelle/HOL proofs for critical algorithms")
     (consequences . ("Mathematically verified" "Increased confidence" "Documentation overhead")))))

(define development-practices
  '((code-style
     (languages . ("Rust" "Julia" "Isabelle" "Nix" "Scheme" "Shell"))
     (formatter . "rustfmt")
     (linter . "clippy"))
    (security
     (sast . "CodeQL")
     (credentials . "env vars only")
     (crypto . "SHAKE256 (SHA3 family)")
     (containers . "Chainguard Wolfi base"))
    (testing
     (coverage-minimum . 70)
     (framework . "cargo test"))
    (versioning
     (scheme . "SemVer 2.0.0"))))

(define design-rationale
  '((why-rsr "RSR ensures consistency, security, and maintainability across the hyperpolymath ecosystem.")
    (why-rust "Memory safety without GC, excellent performance for data processing, strong type system.")
    (why-shake256 "FIPS 202 compliant, extensible output length, part of SHA-3 family.")
    (why-isabelle "Industry-proven theorem prover with excellent Isar proof language.")))

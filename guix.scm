;; SPDX-License-Identifier: MIT OR AGPL-3.0-or-later
;; SPDX-FileCopyrightText: 2024-2025 hyperpolymath
;; ZeroStep - Guix Package Definition
;; Run: guix shell -D -f guix.scm

(use-modules (guix packages)
             (guix gexp)
             (guix git-download)
             (guix build-system cargo)
             ((guix licenses) #:prefix license:)
             (gnu packages base))

(define-public zerostep
  (package
    (name "vae-normalizer")
    (version "1.0.0")
    (source (local-file "." "zerostep-checkout"
                        #:recursive? #t
                        #:select? (git-predicate ".")))
    (build-system cargo-build-system)
    (synopsis "VAE dataset normalizer with formal verification")
    (description "Normalize VAE-decoded image datasets with cryptographic
integrity verification and formal proofs. Supports train/test/val/cal splits,
SHAKE256 checksums, and CUE metadata with Dublin Core.")
    (home-page "https://github.com/hyperpolymath/zerostep")
    ;; Dual licensed: MIT OR AGPL-3.0-or-later
    (license (list license:expat license:agpl3+))))

;; Return package for guix shell
zerostep

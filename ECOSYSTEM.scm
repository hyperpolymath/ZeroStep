;; SPDX-License-Identifier: MIT OR AGPL-3.0-or-later
;; SPDX-FileCopyrightText: 2024-2025 hyperpolymath
;; ECOSYSTEM.scm — zerostep (VAE Dataset Normalizer)

(ecosystem
  (version "1.0.0")
  (name "zerostep")
  (type "project")
  (purpose "Normalize VAE-decoded image datasets with cryptographic integrity
and formal verification for AI image detection model training.")

  (position-in-ecosystem
    "Part of hyperpolymath ecosystem. Follows RSR guidelines for reproducible,
secure, and well-documented software.")

  (related-projects
    (project (name "rhodium-standard-repositories")
             (url "https://github.com/hyperpolymath/rhodium-standard-repositories")
             (relationship "standard"))
    (project (name "VAEDecodedImages-SDXL")
             (url "https://huggingface.co/datasets/joshuajewell/VAEDecodedImages-SDXL")
             (relationship "dataset")))

  (what-this-is "A Rust CLI tool for processing VAE-decoded image datasets:
- SHAKE256 checksums for cryptographic integrity
- Train/test/val/cal splits (random + stratified)
- CUE metadata with Dublin Core
- Isabelle/HOL formal verification proofs
- Julia/Flux training utilities")

  (what-this-is-not "- NOT a VAE model implementation
- NOT a training framework
- NOT exempt from RSR compliance"))

(use-modules
  (guix packages)
  (guix download)
  (guix build-system cargo)
  (guix licenses)
  ((guix licenses) #:prefix license:))

(package
  (name "merkler-static")
  (version "1.0.0")
  (source (local-file "../merkler-static" #:recursive? #t))
  (build-system cargo-build-system)
  (arguments
    `(#:cargo-inputs
      (("rust-sha3" ,rust-sha3-0.10)
       ("rust-serde" ,rust-serde-1)
       ("rust-serde-json" ,rust-serde-json-1)
       ("rust-hex" ,rust-hex-0.4)
       ("rust-rand" ,rust-rand-0.8)
       ("rust-zeroize" ,rust-zeroize-1))
      #:cargo-development-inputs
      (())
      #:features '("biokey")))
  (home-page "https://github.com/robertringler/QRATUM")
  (synopsis "Biokey-enabled Merkle chain builder for VITRA-E0")
  (description
    "Ephemeral biometric cryptographic key system for sovereign genomics.
Provides RAM-only biokey derivation, zero-knowledge proofs, and dual-signature
authorization for whole genome sequencing pipelines.")
  (license license:asl2.0))

merkler-static

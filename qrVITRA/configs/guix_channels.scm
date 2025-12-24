;;; guix_channels.scm --- Pinned Guix channels for deterministic builds
;;;
;;; This file specifies exact commit hashes for all Guix channels used
;;; in VITRA-E0 builds, ensuring bit-identical reproducibility across
;;; different build environments and time periods.
;;;
;;; Usage:
;;;   guix time-machine -C guix_channels.scm -- build -f ../guix/merkler-static.scm

(list
 ;; Main Guix channel - pinned to stable release
 (channel
  (name 'guix)
  (url "https://git.savannah.gnu.org/git/guix.git")
  ;; Pinned to Guix 1.4.0 release (2023-12-19)
  ;; Update this commit to track newer stable releases
  (commit "a0178d34f582b50e9bdbb0403943129ae5b560ff")
  (introduction
   (make-channel-introduction
    "9edb3f66fd807b096b48283debdcddccfea34bad"
    (openpgp-fingerprint
     "BBB0 2DDF 2CEA F6A8 0D1D  E643 A2A0 6DF2 A33A 54FA"))))
 
 ;; Nonguix channel - for NVIDIA drivers and CUDA support
 ;; Required for GPU-accelerated genomics pipelines
 (channel
  (name 'nonguix)
  (url "https://gitlab.com/nonguix/nonguix")
  ;; Pinned to commit compatible with Guix 1.4.0
  (commit "897c1a470da759236cc11798f4e0a5f7d4d59fbc")
  (introduction
   (make-channel-introduction
    "897c1a470da759236cc11798f4e0a5f7d4d59fbc"
    (openpgp-fingerprint
     "2A39 3FFF 68F4 EF7A 3D29  12AF 6F51 20A0 22FB B2D5"))))
 
 ;; Guix-HPC channel - for bioinformatics tools
 ;; Provides optimized builds of genomics software
 (channel
  (name 'guix-hpc)
  (url "https://gitlab.inria.fr/guix-hpc/guix-hpc.git")
  ;; Pinned to recent stable commit
  (commit "2d2eb7d894f7c496f324487854b105273c2e9ede"))
 
 ;; Guix-Science channel - for scientific computing
 ;; Provides additional bioinformatics packages
 (channel
  (name 'guix-science)
  (url "https://github.com/guix-science/guix-science.git")
  ;; Pinned to stable commit
  (commit "a3e7b3e5f8f0ea43ea4a0ec2e8dc1f48e49ff8e3")))

;;; Determinism Notes:
;;;
;;; 1. Never use 'latest' or branch names - always pin to specific commits
;;; 2. Update commits only when intentionally upgrading dependencies
;;; 3. Document reasons for commit changes in git history
;;; 4. Test reproducibility after any channel update
;;; 5. Store channel manifest alongside build artifacts
;;;
;;; Verification:
;;;   guix describe -f channels > actual_channels.scm
;;;   diff guix_channels.scm actual_channels.scm
;;;
;;; Reproducibility Guarantee:
;;;   Same channels.scm + same source = same binary (bit-identical)

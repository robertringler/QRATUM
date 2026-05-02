;; Guix channels configuration for VITRA-E0
;; Provides reproducible build environment

(list
  (channel
    (name 'guix)
    (url "https://git.savannah.gnu.org/git/guix.git")
    (branch "master")
    (commit "INSERT_PINNED_COMMIT_HERE")
    (introduction
      (make-channel-introduction
        "9edb3f66fd807b096b48283debdcddccfea34bad"
        (openpgp-fingerprint
          "BBB0 2DDF 2CEA F6A8 0D1D  E643 A2A0 6DF2 A33A 54FA"))))
  
  (channel
    (name 'guix-science)
    (url "https://github.com/guix-science/guix-science.git")
    (branch "master"))
  
  (channel
    (name 'guix-bioinformatics)
    (url "https://github.com/genenetwork/guix-bioinformatics.git")
    (branch "master")))

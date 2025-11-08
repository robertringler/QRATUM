% Cosmic Echoes of REVULTRA
\version "2.24.0"

melody = \relative c' {
  \tempo 4 = 96
  c4 d8 e f4 g
  a2 g4 f
  e4 d8 c d4 e
  g2. r4
}

curvature = \relative c' {
  c2 e
  g4 f e d
  c1
}

\score {
  \new StaffGroup <<
    \new Staff \with { instrumentName = "Manifold" } { \clef treble \melody }
    \new Staff \with { instrumentName = "Curvature" } { \clef alto \curvature }
  >>
  \layout { }
  \midi { }
}

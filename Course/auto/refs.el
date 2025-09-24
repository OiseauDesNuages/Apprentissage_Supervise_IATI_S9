;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "refs"
 (lambda ()
   (LaTeX-add-bibitems
    "zhang2023dive"
    "prince2023understanding"
    "pml1Book"
    "pml2Book"
    "mitchelllearning"
    "10.5555/3327345.3327535"
    "mlbook2022"
    "azad2023lossfunctionserasemantic"))
 '(or :bibtex :latex))


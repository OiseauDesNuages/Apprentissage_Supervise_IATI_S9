(TeX-add-style-hook
 "ue_apprentissage_supervisé_iati"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("beamer" "pressentation" "10pt" "aspectratio=169" "xcolor=table" "colorlinks=true")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem") ("babel" "english") ("biblatex" "citestyle=verbose" "backend=biber") ("doclicense" "type={CC}" "modifier={by-sa}" "version={4.0}" "")))
   (add-to-list 'LaTeX-verbatim-environments-local "minted")
   (add-to-list 'LaTeX-verbatim-environments-local "semiverbatim")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "beamer"
    "beamer10"
    "inputenc"
    "fontenc"
    "graphicx"
    "longtable"
    "wrapfig"
    "rotating"
    "ulem"
    "amsmath"
    "amssymb"
    "capt-of"
    "overlock"
    "babel"
    "tikz"
    "pgfplots"
    "smartdiagram"
    "csquotes"
    "biblatex"
    "doclicense"
    "minted"
    "tcolorbox")
   (LaTeX-add-labels
    "eq:lin:grad:w"
    "eq:lin:grad:b")
   (LaTeX-add-bibliographies
    "refs")
   (LaTeX-add-tcolorbox-newtcolorboxes
    '("work" "1" "[" "")))
 :latex)


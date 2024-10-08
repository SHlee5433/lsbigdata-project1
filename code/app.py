# This file generated by Quarto; do not edit by hand.
# shiny_mode: core

from __future__ import annotations

from pathlib import Path
from shiny import App, Inputs, Outputs, Session, ui




def server(input: Inputs, output: Outputs, session: Session) -> None:
    import seaborn as sns
    penguins = sns.load_dataset("penguins")

    # ========================================================================

    from shiny.express import render, ui
    ui.input_select("x", "Variable:",
                    choices=["bill_length_mm", "bill_depth_mm"])
    ui.input_select("dist", "Distribution:", choices=["hist", "kde"])
    ui.input_checkbox("rug", "Show rug marks", value = False)

    # ========================================================================

    @render.plot
    def displot():
        sns.displot(
            data=penguins, hue="species", multiple="stack",
            x=input.x(), rug=input.rug(), kind=input.dist())

    # ========================================================================



    return None


_static_assets = ["shiny_files","code\\shiny_files\\libs\\quarto-html\\tippy.css","code\\shiny_files\\libs\\quarto-html\\quarto-syntax-highlighting.css","code\\shiny_files\\libs\\bootstrap\\bootstrap-icons.css","code\\shiny_files\\libs\\bootstrap\\bootstrap.min.css","code\\shiny_files\\libs\\quarto-dashboard\\datatables.min.css","code\\shiny_files\\libs\\clipboard\\clipboard.min.js","code\\shiny_files\\libs\\quarto-html\\quarto.js","code\\shiny_files\\libs\\quarto-html\\popper.min.js","code\\shiny_files\\libs\\quarto-html\\tippy.umd.min.js","code\\shiny_files\\libs\\quarto-html\\anchor.min.js","code\\shiny_files\\libs\\bootstrap\\bootstrap.min.js","code\\shiny_files\\libs\\quarto-dashboard\\quarto-dashboard.js","code\\shiny_files\\libs\\quarto-dashboard\\stickythead.js","code\\shiny_files\\libs\\quarto-dashboard\\datatables.min.js","code\\shiny_files\\libs\\quarto-dashboard\\pdfmake.min.js","code\\shiny_files\\libs\\quarto-dashboard\\vfs_fonts.js","code\\shiny_files\\libs\\quarto-dashboard\\web-components.js","code\\shiny_files\\libs\\quarto-dashboard\\components.js"]
_static_assets = {"/" + sa: Path(__file__).parent / sa for sa in _static_assets}

app = App(
    Path(__file__).parent / "shiny.html",
    server,
    static_assets=_static_assets,
)

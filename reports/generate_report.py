from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from pathlib import Path


def build_pdf_report(text, filename="outputs/reports/research_report.pdf"):

    # =========================
    # ENSURE DIRECTORY EXISTS
    # =========================
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    # =========================
    # BUILD PDF
    # =========================
    doc = SimpleDocTemplate(str(path))
    styles = getSampleStyleSheet()

    story = []

    for line in text.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 6))

    doc.build(story)

    print(f"\n📄 Report saved to {path}")

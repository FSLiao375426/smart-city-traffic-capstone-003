"""Build the two-page Part 2 report with ReportLab (authoring dependency)."""
import csv
import json
import logging
from pathlib import Path
import time

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def build_report():
    processed = ROOT / "data/processed"
    quality = json.loads((processed / "quality_report.json").read_text())
    scale = json.loads((processed / "scaling_values.json").read_text())
    quartiles = json.loads((processed / "quartile_values.json").read_text())
    figures = json.loads((ROOT / "part2_python/figures/figure_metadata.json").read_text())
    acceptance = json.loads((ROOT / "part2_python/validation/cli_acceptance.json").read_text())
    with (processed / "traffic_model_features.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = len(reader.fieldnames)
        rows = sum(1 for _ in reader)
    if (rows, columns) != (40575, 36):
        raise ValueError("Report acceptance shape differs from the verified data")
    folder = ROOT / "output/pdf"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / "part2_traffic_pipeline_report.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=22, leading=26,
                              textColor=colors.HexColor("#17364C"), spaceAfter=9))
    styles.add(ParagraphStyle(name="Deck", fontSize=10, leading=14, textColor=colors.HexColor("#506070"), spaceAfter=16))
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 14
    styles["BodyText"].spaceAfter = 8
    styles["Heading2"].fontSize = 12
    styles["Heading2"].textColor = colors.HexColor("#176B9A")
    story = []

    def paragraph(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))

    def table(headers, values, widths):
        data = [[Paragraph(str(cell), styles["BodyText"]) for cell in row] for row in [headers, *values]]
        item = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0F5")),
                                 ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                                 ("TOPPADDING", (0, 0), (-1, -1), 6),
                                 ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#D6E0E8"))]))
        story.extend([item, Spacer(1, 10)])

    paragraph("Reproducible Traffic Analytics Pipeline", "ReportTitle")
    paragraph("NUS AI ML Capstone 003 | Part 2 | Methods and results | 24 November 2026", "Deck")
    paragraph("The workflow converts the supplied traffic and weather records into auditable hourly data, fitted features, three descriptive figures and a three-command analytics application. Raw observations remain unchanged; preprocessing parameters and execution events are saved for review.")
    paragraph("Data preparation and validation", "Heading2")
    paragraph(f"The source contains {quality['raw_rows']:,} rows and nine columns. Schema, timestamps and same-hour traffic consistency are checked before processing. Seventeen exact duplicates are removed. Multiple weather records are retained in a separate detail table while traffic is counted once per observed hour, yielding {rows:,} hourly rows.")
    paragraph("Ten temperatures at 0 K and one abnormal rainfall observation are treated as invalid and filled using reference-period monthly medians, with imputation flags preserved. Both zero-traffic hours are retained because their causes are unverified. Nine complete months are missing (September 2014 to May 2015); missing hours are not generated or interpreted as zero traffic.")
    paragraph("Feature engineering and temporal isolation", "Heading2")
    paragraph(f"Calendar features are hour, day of week and weekend status. Hour sine/cosine coordinates preserve the 24-hour cycle; existing multi-label weather indicators are retained. The demonstration reference period ends at {scale['fit_end']} inclusive ({scale['reference_hours']:,} hours). The remaining 15,246 hours use the same fitted parameters.")
    table(["Scaled variable", "Reference mean", "Population SD"], [
        ["Temperature (K)", f"{scale['columns']['temp']['mean']:.5f}", f"{scale['columns']['temp']['std']:.5f}"],
        ["Cloud cover (%)", f"{scale['columns']['clouds_all']['mean']:.5f}", f"{scale['columns']['clouds_all']['std']:.5f}"],
    ], [180, 140, 175])
    paragraph("Z-scores use (value - reference mean) / population SD (ddof=0). Saved reference-period means and SDs agree with 0 and 1 within 1e-8. All prior columns are preserved: time, scaled and labelled tables contain 33, 35 and 36 columns respectively.")
    q = quartiles["thresholds"]
    paragraph(f"Linear reference-period quantiles define levels 1 to 4: x &lt;= {q['q1']:.0f}; {q['q1']:.0f} &lt; x &lt;= {q['q2']:.0f}; {q['q2']:.0f} &lt; x &lt;= {q['q3']:.0f}; and x &gt; {q['q3']:.0f}. Ties stay together, so groups need not be equal. These traffic-volume proxy labels do not measure observed accident risk.")
    story.append(PageBreak())
    paragraph("Findings and Reproduction", "ReportTitle")
    paragraph("Full-period descriptive findings and verified software behaviour", "Deck")
    table(["Saved Matplotlib figure", "Interpretation"], [
        ["01_hourly_profile.png", "Weekday traffic peaks at 16:00 (6,240.88 vehicles/hour), with another peak at 07:00 (6,061.16). Weekend traffic peaks at 13:00 (4,407.61). Means use observed hours; seasonal composition remains unadjusted."],
        ["02_traffic_distribution.png", "Bins of 250 vehicles show low- and high-volume concentrations. Mean traffic is 3,290.65 vehicles/hour and the median is 3,427. Both zero-traffic observations are included."],
        ["03_temperature_traffic.png", f"All {rows:,} hours are plotted without sampling. Temperature has a weak overall linear association with traffic (Pearson r={figures['pearson_temp_traffic']:.3f}). Time of day and season may confound this association; causation is not established."],
    ], [165, 330])
    paragraph("Reproduction and the mini application", "Heading2")
    paragraph("Use Python 3.11 and the pinned NumPy, pandas and Matplotlib dependencies. From the repository or extracted package root, run <font name='Courier'>python -m part2_python.pipeline --all</font>. It regenerates cleaning outputs, all feature tables and three figures. Supporting figure CSVs and source hashes allow independent numerical review.")
    paragraph("The CLI supports <font name='Courier'>at --time</font> for exact-hour traffic/weather, <font name='Courier'>summary --start --end</font> for date-range statistics and quartile counts, and <font name='Courier'>compare --start --end</font> for weekday/weekend means. Invoke each with <font name='Courier'>python -m part2_python.cli</font>. Dates include both endpoints. January 2017 returns 744 hours and mean traffic 3,120.26; the missing January 2015 returns no_data and a null mean.")
    paragraph("Logging and validation", "Heading2")
    paragraph(f"Operational modules use logging.getLogger(__name__). Only the entry point configures console and file handlers, with UTC timestamp, level, module and message. INFO records milestones; WARNING records recoverable cleaning or missing-data events; DEBUG holds fitted diagnostics; ERROR records failures. Integrated events remain in pipeline.log. CLI answers use stdout; status uses stderr and cli.log. No internal progress print() calls are used.")
    paragraph(f"Sixteen unit tests passed. All {len(acceptance['cases'])} saved CLI cases passed: three commands, a no-data result and three invalid-input cases. Malformed dates, reversed ranges and unknown commands return code 2 with clear ERROR messages and no traceback. The full raw-to-figures pipeline passed with warnings treated as errors. Source integrity, prior-column preservation, boundaries and holdout isolation were checked; these checks are not predictive performance estimates.")
    paragraph("Limitations and next steps", "Heading2")
    paragraph("The final modelling split and prediction-time feature availability remain undecided. Refit all learned preprocessing and thresholds on the final training period. Exclude traffic_volume and its derived label from classification inputs. Full-period plots are exploratory, not held-out evaluation. The data cover one corridor and contain no observed accident outcomes; future recommendations should focus on departure timing and disclose these limitations.")
    paragraph("Data: Hogue (2019), Metro Interstate Traffic Volume, UCI Machine Learning Repository, DOI 10.24432/C5X60B (CC BY 4.0). Evidence: processed JSON metadata, figures/, validation/cli_acceptance.json and pipeline.log.", "Deck")

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#D6E0E8"))
        canvas.line(45, 38, A4[0] - 45, 38)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#506070"))
        canvas.drawString(45, 26, "Smart City Traffic Intelligence | Part 2")
        canvas.drawRightString(A4[0] - 45, 26, str(doc.page))

    SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=45, leftMargin=45,
                      topMargin=40, bottomMargin=50, title="Part 2 Traffic Analytics Pipeline",
                      author="Smart City Traffic Capstone").build(story, onFirstPage=footer, onLaterPages=footer)
    logger.info("Saved report %s", destination)
    return destination


if __name__ == "__main__":
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    formatter.converter = time.gmtime
    handlers = [logging.StreamHandler(), logging.FileHandler(ROOT / "part2_python/build_report.log", mode="w", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    build_report()

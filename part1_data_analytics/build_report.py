"""Generate the two-page Part 1 insights report with optional ReportLab tooling."""
import csv
import logging
from pathlib import Path
import time

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def build_report():
    folder = ROOT / "part1_data_analytics/results"

    def read(name):
        with (folder / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    stats = {row["metric"]: float(row["value"]) for row in read("descriptive_statistics.csv")}
    r = float(read("temperature_correlation.csv")[0]["pearson_r"])
    weather = read("weather_average_traffic.csv")
    highest = max(weather, key=lambda row: float(row["mean_traffic"]))
    lowest = min(weather, key=lambda row: float(row["mean_traffic"]))
    difference = float(highest["mean_traffic"]) - float(lowest["mean_traffic"])
    temp_range = {float(row["quantile"]): float(row["temperature_c"]) for row in read("congested_temperature_range.csv")}
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleReport", fontSize=22, leading=26, fontName="Helvetica-Bold", spaceAfter=10,
                              textColor=colors.HexColor("#17364C")))
    styles.add(ParagraphStyle(name="Deck", fontSize=10, leading=14, spaceAfter=14, textColor=colors.HexColor("#506070")))
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 12
    styles["BodyText"].spaceAfter = 6
    styles["Heading2"].spaceBefore = 7
    styles["Heading2"].spaceAfter = 5
    styles["Heading2"].fontSize = 12
    styles["Heading2"].textColor = colors.HexColor("#176B9A")
    story = []

    def p(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))

    def table(headers, rows, widths):
        data = [[Paragraph(str(cell), styles["BodyText"]) for cell in row] for row in [headers, *rows]]
        item = Table(data, colWidths=widths)
        item.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.HexColor("#E8F0F5")),
                                 ("VALIGN",(0,0),(-1,-1),"TOP"),
                                 ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
                                 ("LINEBELOW",(0,0),(-1,-1),.3,colors.HexColor("#D6E0E8"))]))
        story.extend([item, Spacer(1,10)])

    p("Traffic Analytics Insights", "TitleReport")
    p("For the smart-city mobility team | NUS AI ML Capstone 003 Part 1 | 24 November 2026", "Deck")
    p("Prioritise monitoring by time of day, retain weather as context and audit observation coverage before interpreting annual change. The dataset supports descriptions of one I-94 corridor; it does not establish weather effects on traffic or accident risk.")
    p("Scope and data quality", "Heading2")
    p("Analysis uses 40,575 unique observed hours from October 2012 to September 2018. The 48,204 raw rows include multiple weather descriptions per hour; summing raw rows would overcount traffic. Seventeen exact duplicates are removed in cleaning. Two zero-traffic hours are retained; their causes are unknown. Nine complete months from September 2014 to May 2015 are absent. Missing hours are not replaced with zeros.")
    table(["Hourly traffic statistic", "Value"], [
        ["Mean / median", f"{stats['mean']:,.2f} / {stats['median']:,.0f} vehicles/hour"],
        ["Population SD / variance (ddof=0)", f"{stats['population_std_ddof0']:,.2f} vehicles/hour / {stats['population_variance_ddof0']:,.2f} (vehicles/hour)<super>2</super>"],
        ["Minimum / maximum / range", "0 / 7,280 / 7,280 vehicles/hour"],
    ], [240,255])
    p("The large standard deviation and range show substantial variation around the overall mean. Aggregating all hours hides distinct low- and high-volume periods. Power Query's sample SD (1,984.77; N-1 denominator) differs slightly from the descriptive population SD above; both conventions are documented.")
    p("Annual totals require a coverage check", "Heading2")
    p("From 2013 to 2014, recorded traffic totals fell 39.03%, while observed hours fell 38.29% and mean traffic per observed hour fell 1.19%. Full-year coverage declined from 83.26% to 51.38%. Most of the total reduction accompanies fewer recorded hours; it is not evidence of a comparable demand collapse.")
    p("From 2016 to 2017, observed totals rose 17.53%, observed hours rose 11.16% and mean hourly traffic rose 5.73%. Coverage improved from 89.23% to 99.46%. Review month and hour composition before attributing the increase to underlying demand. The 2012 record begins in October and is not a full-year baseline.")
    p("Time and holiday findings", "Heading2")
    p("In 2017, the highest mean hourly traffic was at 16:00 (5,834.37 vehicles/hour); the lowest was at 03:00 (385.81). Prioritise time-specific monitoring. Calendar New Year's Day 2015 has no observations; 2016 has only 18 hours. Its observed mean temperature was -6.06 C, versus -3.06 C on 1 January 2017. The 2017 source holiday label occurs on 2 January (-1.30 C); keep these date definitions separate.")
    story.append(PageBreak())
    p("Weather Evidence and Actions", "TitleReport")
    p("Distinguish observed associations from causal and predictive claims", "Deck")
    p("Probability analysis", "Heading2")
    table(["Event or conditional probability", "Observed result"], [
        ["Congestion: traffic volume >5,500", "6,107 / 40,575 = 15.05%"],
        ["Clear weather", "13,371 / 40,575 = 32.95%"],
        ["Congestion AND clear weather", "1,763 / 40,575 = 4.35%"],
        ["Clear weather given congestion", "1,763 / 6,107 = 28.87%"],
        ["Temperature >292 K given congestion", "1,699 / 6,107 = 27.82%"],
    ], [325,170])
    p("The observed joint probability (4.3450%) differs from P(Clear) x P(Congestion) (4.9599%). The empirical equality for independence does not hold. This is a descriptive comparison, not a formal inferential test; consecutive traffic hours are not independent random draws.")
    p("The crude clear-versus-cloudy congestion odds ratio is 0.7361. This compares Clear without Clouds (1,762 congested; 11,602 not) with Clouds without Clear (2,586; 12,534). Seven hours with both labels and 12,084 with neither are excluded. Other weather labels may coexist within either comparison group.")
    p("Time composition changes the comparison", "Heading2")
    p("Unadjusted congestion rates are 13.18% for the clear group and 17.10% for the cloudy group. Using the same pooled 24-hour distribution for both groups gives 15.28% and 15.23% (difference +0.05 percentage points). Hour x weekday/weekend adjustment gives 15.65% and 14.99%. Time composition substantially reduces the crude gap. These descriptive adjustments do not estimate a weather treatment effect or its incremental model value.")
    p(f"Across nonexclusive weather categories, {highest['weather']} has the highest mean traffic ({float(highest['mean_traffic']):,.2f}; {int(highest['observed_hours']):,} hours) and {lowest['weather']} the lowest ({float(lowest['mean_traffic']):,.2f}; {int(lowest['observed_hours'])} hours), a difference of {difference:,.2f}. The four-hour Squall sample is too small to support a dependable operational ranking. Category counts overlap and must not be summed as unique hours.")
    p(f"Temperature has a weak positive overall linear correlation with traffic (r={r:.4f}, 40,565 directly observed temperature pairs); including ten imputed hours gives r=0.1395. Congested hours span {temp_range[0]:.2f} to {temp_range[1]:.2f} C, with the middle 80% between {temp_range[.1]:.2f} and {temp_range[.9]:.2f} C. This is a descriptive range, not a causal cutoff. Time of day and season may confound the association. The maximum hourly flow (7,280) occurred at -2.40 C; high traffic is not confined to warm conditions.")
    p("Recommended use and dashboard", "Heading2")
    p("Use hour and weekday/weekend patterns to guide monitoring, maintain weather as contextual information, and audit missing periods and sensor anomalies before operational decisions. Evaluate any added predictive value on future time periods before deployment. No accident outcomes are supplied; congestion is a traffic-volume event, not confirmed accident risk.")
    p("The Power BI deliverable contains daily, hourly, weather and temperature visuals; three KPI cards; and hour, weather and traffic-category slicers. Its categories are Low &lt;4,500, Medium 4,500-5,500 inclusive, and High &gt;5,500. These differ from the Part 2 reference-fitted quartile labels. Dashboard interaction was previously checked by the user; current file structure was independently verified.")
    p("Evidence: sql/, results/, traffic.sqlite and traffic_dashboard.pbix in part1_data_analytics. Data: Hogue (2019), Metro Interstate Traffic Volume, UCI; DOI 10.24432/C5X60B (CC BY 4.0).", "Deck")
    output = ROOT / "output/pdf/part1_data_analytics_insights_report.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)

    def footer(canvas, doc):
        canvas.setFont("Helvetica",8)
        canvas.setFillColor(colors.HexColor("#506070"))
        canvas.drawString(45,25,"Smart City Traffic Intelligence | Part 1")
        canvas.drawRightString(A4[0]-45,25,str(doc.page))

    SimpleDocTemplate(str(output),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=35,bottomMargin=43,
                      title="Part 1 Data Analytics Insights Report").build(story,onFirstPage=footer,onLaterPages=footer)
    logger.info("Saved %s",output)


if __name__ == "__main__":
    formatter=logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",datefmt="%Y-%m-%dT%H:%M:%SZ")
    formatter.converter=time.gmtime
    handlers=[logging.StreamHandler(),logging.FileHandler(ROOT / "part1_data_analytics/build_report.log",mode="w",encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO,handlers=handlers)
    build_report()

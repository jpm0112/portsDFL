"""
Generate PDF documentation files for the BAP service time project.

Creates:
  - data/column_description.pdf
  - data/project_description.pdf

Uses reportlab for PDF generation with clean formatting.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch                 # points in one inch (multiply to size things)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    # Flowables — building blocks that reportlab stacks down the page.
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)


# --- Color palette ---
DARK_BLUE = HexColor("#1a3a5c")
MEDIUM_BLUE = HexColor("#2c5f8a")
LIGHT_BLUE = HexColor("#e8f0f8")
HEADER_BG = HexColor("#2c5f8a")
ROW_ALT = HexColor("#f5f8fc")
WHITE = HexColor("#ffffff")
DARK_GRAY = HexColor("#333333")
LIGHT_GRAY = HexColor("#cccccc")

# Path to the sibling 'data/' folder, regardless of where the script is run from.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def get_styles():
    """Build custom paragraph styles for the PDF."""
    # Start from reportlab's built-in styles, then add customized ones on top.
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "DocTitle", parent=styles["Title"],
        fontSize=20, textColor=DARK_BLUE, spaceAfter=6, alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"],
        fontSize=11, textColor=MEDIUM_BLUE, alignment=TA_CENTER, spaceAfter=20
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading1"],
        fontSize=14, textColor=DARK_BLUE, spaceBefore=18, spaceAfter=8,
        borderWidth=0, borderPadding=0
    ))
    styles.add(ParagraphStyle(
        "SubHead", parent=styles["Heading2"],
        fontSize=11, textColor=MEDIUM_BLUE, spaceBefore=12, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        "BodyText2", parent=styles["Normal"],
        fontSize=9, textColor=DARK_GRAY, spaceAfter=6, leading=13
    ))
    styles.add(ParagraphStyle(
        "SmallText", parent=styles["Normal"],
        fontSize=8, textColor=DARK_GRAY, leading=11
    ))
    styles.add(ParagraphStyle(
        "CodeBlock", parent=styles["Normal"],
        fontSize=7.5, fontName="Courier", textColor=DARK_GRAY,
        backColor=HexColor("#f0f0f0"), leading=10, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        "TableHeader", parent=styles["Normal"],
        fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, leading=10
    ))
    styles.add(ParagraphStyle(
        "TableCell", parent=styles["Normal"],
        fontSize=7.5, textColor=DARK_GRAY, leading=10
    ))
    return styles


# col_widths=None lets reportlab auto-size the columns.
def make_table(headers, rows, col_widths=None):
    """
    Build a styled table with header row and alternating row colors.

    Input:  headers (list of str), rows (list of lists), optional col_widths
    Output: Table flowable
    """
    data = [headers] + rows

    # repeatRows=1 re-prints the header row atop each new page if the table spills over.
    table = Table(data, colWidths=col_widths, repeatRows=1)
    # Each tuple is one styling command; the two (col, row) pairs mark the
    # top-left and bottom-right corners of the cell range it applies to (-1 = last).
    style_cmds = [
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        # Body
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 7),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_GRAY),
        # Grid
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    # Alternating row colors (zebra), skipping the header at row 0.
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))

    table.setStyle(TableStyle(style_cmds))
    return table


def hr():
    """Horizontal rule flowable."""
    return HRFlowable(width="100%", thickness=0.5, color=LIGHT_GRAY,
                      spaceBefore=6, spaceAfter=6)


def build_column_description_pdf():
    """Generate column_description.pdf with all column definitions."""
    filepath = os.path.join(DATA_DIR, "column_description.pdf")
    doc = SimpleDocTemplate(filepath, pagesize=letter,
                            topMargin=0.6*inch, bottomMargin=0.6*inch,
                            leftMargin=0.6*inch, rightMargin=0.6*inch)
    styles = get_styles()
    # 'story' is the running list of flowables; doc.build(story) lays it out at the end.
    story = []

    # Title
    story.append(Paragraph("Column Description", styles["DocTitle"]))
    story.append(Paragraph(
        "BAP Service Time Prediction \u2014 Puerto de San Antonio, Chile",
        styles["DocSubtitle"]
    ))
    story.append(Paragraph(
        "Dataset: training_dataset.csv | 5,597 rows | 44 columns",
        styles["DocSubtitle"]
    ))
    story.append(hr())

    W = 7.3 * inch  # total usable width (defined for reference; not used below)

    # --- Target Variables ---
    # The \u00xx escapes embed accented characters so the file stays plain ASCII on disk.
    story.append(Paragraph("1. Target Variables / Variables Objetivo", styles["SectionHead"]))
    story.append(Paragraph(
        "Two service time metrics computed from timestamps. Not derived from Excel formulas.",
        styles["BodyText2"]
    ))
    # Column widths (must match the number of headers/cells per row).
    cols = [1.5*inch, 1.2*inch, 2.4*inch, 2.2*inch]
    story.append(make_table(
        ["Column", "Spanish", "Description", "Calculation"],
        [
            ["estadia_sitio_hours", u"Estad\u00eda sitio",
             "Berth occupation time (hours). First mooring to last unmooring.",
             u"\u00daltima esp\u00eda desatraque \u2212 1era esp\u00eda atraque"],
            ["tiempo_en_puerto_hours", "Tiempo en puerto",
             "Time inside port (hours). Pilot boarding to departure.",
             u"Zarpe \u2212 Fecha pr\u00e1ctico atraque"],
        ],
        col_widths=cols
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Note:</b> F. arribo records arrival at the anchorage (outside the port), not port entry. "
        "Pilot boarding marks actual port entry. Dry bulk vessels wait a median of 207h (~8.6 days) at anchor.",
        styles["SmallText"]
    ))

    # --- Direct Features ---
    story.append(Paragraph("2. Direct Features / Caracter\u00edsticas Directas", styles["SectionHead"]))
    story.append(Paragraph(
        "Pre-berthing information only. No departure-time data or berth assignment (BAP decision variable).",
        styles["BodyText2"]
    ))
    cols = [1.6*inch, 1.2*inch, 2.2*inch, 2.3*inch]
    story.append(make_table(
        ["Column", "Spanish Original", "Description", "Calculation"],
        [
            ["vessel_code", u"C\u00f3d. nave", "Unique vessel identifier (IMO)", "Direct"],
            ["vessel_name", "Nave", "Vessel name / Nombre de la nave", "Direct"],
            ["vessel_type", "T. nave", "Original vessel type (16 categories)", "Direct"],
            ["vessel_type_group", "T. nave (agrupado)", "Aggregated vessel type (7 groups)", "Mapped from vessel_type"],
            ["trg", "TRG", "Gross registered tonnage / Tonelaje bruto", "Direct"],
            ["terminal", "Terminal", "Port terminal (DP World, STI, PANUL, QC, EPSA)", "Direct"],
            ["agency", "Agencia", "Maritime agency name / Agencia mar\u00edtima", 'Split "RUT - NAME" on " - ", take name'],
            ["shipping_line", "L. naviera", 'Shipping line. "UNKNOWN" if missing (30%)', "Direct, NaN \u2192 UNKNOWN"],
            ["service_route", "Servicio", 'Service route. "UNKNOWN" if missing (30%)', "Direct, NaN \u2192 UNKNOWN"],
            ["origin_port", "Puerto origen", "Port of origin / Puerto de origen", "Direct"],
            ["dest_port", "Puerto destino", "Port of destination / Puerto de destino", "Direct"],
            ["origin_region", "Puerto origen (regi\u00f3n)", "Geographic region of origin (14 regions)", "Mapped via port_regions.py"],
            ["dest_region", "Puerto destino (regi\u00f3n)", "Geographic region of destination", "Mapped via port_regions.py"],
            ["draft_arrival_bow", "C. arribo proa", "Bow draft on arrival (meters)", "Direct"],
            ["draft_arrival_stern", "C. arribo popa", "Stern draft on arrival (meters)", "Direct"],
            ["draft_arrival_mean", "Calado medio arribo", "Mean arrival draft (meters)", "(proa + popa) / 2"],
            ["max_arrival_draft", u"Calado m\u00e1ximo arribo", "Maximum arrival draft (meters)", "max(proa, popa)"],
            ["draft_trim_arrival", "Trimado al arribo", "Arrival trim (m). Positive = stern-heavy", "popa - proa"],
        ],
        col_widths=cols
    ))

    story.append(PageBreak())

    # --- Temporal Features ---
    story.append(Paragraph("3. Temporal Features / Caracter\u00edsticas Temporales", styles["SectionHead"]))
    story.append(Paragraph(
        "Derived from F. arribo (arrival at anchorage datetime).",
        styles["BodyText2"]
    ))
    cols = [1.5*inch, 1.3*inch, 2.3*inch, 2.2*inch]
    story.append(make_table(
        ["Column", "Spanish", "Description", "Calculation"],
        [
            ["arrival_month", "Mes de arribo", "Month of arrival (1-12)", "F. arribo.month"],
            ["arrival_day_of_week", u"D\u00eda de la semana", "Day of week (0=Mon, 6=Sun)", "F. arribo.weekday()"],
            ["arrival_hour", "Hora de arribo", "Hour of arrival (0-23)", "F. arribo.hour"],
            ["arrival_year", u"A\u00f1o de arribo", "Year of arrival", "F. arribo.year"],
            ["quarter", "Trimestre", "Quarter of the year (1-4)", "(month - 1) // 3 + 1"],
            ["is_weekend_arrival", "Fin de semana", "Weekend arrival flag (0/1)", "1 if day_of_week >= 5"],
        ],
        col_widths=cols
    ))

    # --- Historical Features ---
    story.append(Paragraph("4. Historical Vessel Features / Caracter\u00edsticas Hist\u00f3ricas", styles["SectionHead"]))
    story.append(Paragraph(
        "Expanding-window statistics over prior visits of the same vessel, sorted chronologically "
        "by first mooring time. Shifted by 1 to exclude the current visit (no data leakage). "
        "NaN for first visits (1,550 vessels).",
        styles["BodyText2"]
    ))
    cols = [1.9*inch, 1.6*inch, 3.8*inch]
    story.append(make_table(
        ["Column", "Spanish", "Calculation"],
        [
            ["vessel_avg_berth_stay", u"Promedio estad\u00eda nave",
             "groupby(vessel_code)[estadia_sitio].expanding().mean().shift(1)"],
            ["vessel_median_berth_stay", u"Mediana estad\u00eda nave",
             "groupby(vessel_code)[estadia_sitio].expanding().median().shift(1)"],
            ["vessel_std_berth_stay", u"Desv. est. estad\u00eda nave",
             "groupby(vessel_code)[estadia_sitio].expanding().std().shift(1)"],
            ["vessel_visit_count", "Visitas previas nave",
             "groupby(vessel_code).cumcount() (0 = first visit)"],
            ["vessel_last_berth_stay", u"\u00daltima estad\u00eda nave",
             "groupby(vessel_code)[estadia_sitio].shift(1)"],
            ["vessel_avg_berth_stay_at_terminal", "Promedio nave en terminal",
             "groupby([vessel_code, terminal])[estadia_sitio].expanding().mean().shift(1)"],
            ["vessel_visit_count_at_terminal", "Visitas nave en terminal",
             "groupby([vessel_code, terminal]).cumcount()"],
        ],
        col_widths=cols
    ))

    # --- Group-Level Features ---
    story.append(Paragraph("5. Group-Level Features / Caracter\u00edsticas de Grupo", styles["SectionHead"]))
    story.append(Paragraph(
        "Expanding-window averages at higher grouping levels, shifted by 1. "
        "Useful as fallback predictions when vessel-specific history is unavailable.",
        styles["BodyText2"]
    ))
    cols = [1.9*inch, 1.6*inch, 3.8*inch]
    story.append(make_table(
        ["Column", "Spanish", "Calculation"],
        [
            ["type_terminal_avg_stay", "Promedio tipo+terminal",
             "groupby([vessel_type_group, terminal])[estadia_sitio].expanding().mean().shift(1)"],
            ["type_avg_stay", "Promedio tipo",
             "groupby(vessel_type_group)[estadia_sitio].expanding().mean().shift(1)"],
            ["terminal_avg_stay", "Promedio terminal",
             "groupby(terminal)[estadia_sitio].expanding().mean().shift(1)"],
        ],
        col_widths=cols
    ))

    # --- Reference Columns ---
    story.append(Paragraph("6. Reference Columns / Columnas de Referencia", styles["SectionHead"]))
    story.append(Paragraph(
        "Not intended as model features. Included for traceability and analysis.",
        styles["BodyText2"]
    ))
    cols = [1.6*inch, 1.4*inch, 2.1*inch, 2.2*inch]
    story.append(make_table(
        ["Column", "Spanish Original", "Description", "Calculation"],
        [
            ["arrival_datetime", "F. arribo", "Arrival at anchorage", "Direct"],
            ["pilot_boarding_datetime", u"Fecha pr\u00e1ctico atraque", "Pilot boarding (port entry)", "Direct"],
            ["first_mooring_datetime", u"1era esp\u00eda atraque", "First mooring line secured", "Direct"],
            ["last_unmooring_datetime", u"\u00daltima esp\u00eda desatraque", "Last unmooring line cast off", "Direct"],
            ["departure_datetime", "Zarpe", "Vessel departure", "Direct"],
            ["espera_preatraque_hours", "Espera pre-atraque", "Anchorage wait time (hours)", "(Zarpe - F.arribo) - estadia_sitio"],
            ["quality_flag", "Bandera calidad", "1 if dispatch < reception (12 records)", "Timestamp comparison"],
            ["split", u"Partici\u00f3n", "train / validation / test", "Based on arrival date thresholds"],
        ],
        col_widths=cols
    ))

    story.append(PageBreak())

    # --- Vessel Type Aggregation ---
    story.append(Paragraph(u"7. Vessel Type Aggregation / Agrupaci\u00f3n de Tipos", styles["SectionHead"]))
    cols = [1.3*inch, 3.5*inch, 0.8*inch, 1.0*inch]
    story.append(make_table(
        ["Group", "Original Types / Tipos Originales", "Count", "Avg Berth (h)"],
        [
            ["Container", "Contenedor", "3,066", "36"],
            ["Dry Bulk", "Carga Seca Granel, Mineral/Granel/Petrolero", "951", "74"],
            ["Vehicle Carrier", "Autero, Autotrasbordo", "749", "36"],
            ["Liquid Bulk", "Transp. Quimico, Transp. Liquido, Transp. Asfalto, Petrolero", "587", "22"],
            ["General Cargo", "Tradicional, Carga de Proyecto, Chipero, Refrigerado, Otros", "171", "36"],
            ["Passenger", "Pasajeros", "80", "19"],
            ["Other", "Nave Armada", "1", "65"],
        ],
        col_widths=cols
    ))

    # --- Port Regions ---
    story.append(Paragraph("8. Port Regions / Regiones Portuarias", styles["SectionHead"]))
    cols = [1.3*inch, 5.0*inch]
    story.append(make_table(
        [u"Regi\u00f3n / Region", "Examples / Ejemplos"],
        [
            ["Chilean_North", "Arica, Iquique, Antofagasta, Mejillones, Coquimbo"],
            ["Chilean_Central", "Valparaiso, San Antonio, Coronel, Lirquen, San Vicente"],
            ["Chilean_South", "Puerto Montt, Punta Arenas, Chacabuco"],
            ["Peru", "Callao, Ilo, Matarani, Paita"],
            ["Brazil", "Santos, Rio Grande, Paranagua, Rio de Janeiro"],
            ["Argentina", "Buenos Aires, Rosario, Zarate, Bahia Blanca, Montevideo"],
            ["Ecuador_Colombia", "Guayaquil, Buenaventura, Manta"],
            ["Central_America", "Balboa, Cristobal, Manzanillo, Lazaro Cardenas"],
            ["North_America", "Houston, Long Beach, Los Angeles, Vancouver"],
            ["Asia", "Shanghai, Busan, Singapore, Hong Kong, Yokohama"],
            ["Europe", "Rotterdam, Zeebrugge, London, Livorno"],
            ["Africa", "Durban, Capetown, Port Elizabeth"],
            ["Oceania", "Melbourne, Fremantle, Tauranga"],
            ["Other", "Unmapped ports / Puertos no mapeados"],
        ],
        col_widths=cols
    ))

    doc.build(story)
    print(f"  Created {filepath}")


def build_project_description_pdf():
    """Generate project_description.pdf with project overview."""
    filepath = os.path.join(DATA_DIR, "project_description.pdf")
    doc = SimpleDocTemplate(filepath, pagesize=letter,
                            topMargin=0.6*inch, bottomMargin=0.6*inch,
                            leftMargin=0.7*inch, rightMargin=0.7*inch)
    styles = get_styles()
    story = []

    W = 7.1 * inch  # usable width reference (defined but not used below)

    # Title
    story.append(Paragraph("Project Description", styles["DocTitle"]))
    story.append(Paragraph(
        "BAP Service Time Prediction \u2014 Puerto de San Antonio, Chile",
        styles["DocSubtitle"]
    ))
    story.append(hr())

    # Objective
    story.append(Paragraph("1. Objective", styles["SectionHead"]))
    story.append(Paragraph(
        "Predict the service time of vessels at Puerto de San Antonio to support a "
        "Berth Allocation Problem (BAP) optimizer. The port has 5 terminals (DP World, STI, "
        "PANUL, QC, EPSA) that independently plan berth schedules but share a common port "
        "entry channel. Accurate service time predictions enable better berth scheduling and "
        "port entry coordination.",
        styles["BodyText2"]
    ))

    # Problem Context
    story.append(Paragraph("2. Problem Context", styles["SectionHead"]))
    story.append(Paragraph(
        "The BAP assigns incoming vessels to available berths while minimizing waiting times "
        "and maximizing utilization. A key input is the expected service time: how long each "
        "vessel will occupy a berth. This project builds a training dataset to develop a "
        "predictive model for that service time, using only information available before "
        "the vessel berths.",
        styles["BodyText2"]
    ))

    story.append(Paragraph("Port Operations Sequence", styles["SubHead"]))
    steps = [
        ("1. Arrival at anchorage", "F. arribo \u2014 vessel arrives and anchors outside the port"),
        ("2. Waiting at anchorage", "Vessel waits for berth availability (can be days for bulk carriers)"),
        ("3. Pilot boarding", u"Fecha pr\u00e1ctico atraque \u2014 pilot boards, vessel enters port channel"),
        ("4. Mooring", u"1era esp\u00eda atraque to \u00daltima esp\u00eda atraque \u2014 ~42 min average"),
        ("5. Cargo operations", "Loading/unloading at berth"),
        ("6. Unmooring", u"1era esp\u00eda desatraque to \u00daltima esp\u00eda desatraque"),
        ("7. Departure", "Zarpe \u2014 vessel exits the port"),
    ]
    for step, desc in steps:
        story.append(Paragraph(
            f"<b>{step}</b>: {desc}", styles["SmallText"]
        ))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Key insight:</b> F. arribo records arrival at the anchorage (outside the port), "
        "not entry into the port. Dry bulk vessels wait a median of 207 hours (~8.6 days) at "
        "anchor before the pilot boards. The pilot-to-mooring gap is a consistent ~42 minutes "
        "across all vessel types, confirming pilot boarding marks actual port entry.",
        styles["BodyText2"]
    ))

    # Target Variables
    story.append(Paragraph("3. Target Variables", styles["SectionHead"]))
    cols = [1.6*inch, 2.5*inch, 3.0*inch]
    story.append(make_table(
        ["Target", "Formula", "Use Case"],
        [
            ["estadia_sitio_hours\n(Berth stay)",
             u"\u00daltima esp\u00eda desatraque\n\u2212 1era esp\u00eda atraque",
             "Berth allocation: how long the berth is occupied"],
            ["tiempo_en_puerto_hours\n(Port time)",
             u"Zarpe \u2212 Fecha pr\u00e1ctico atraque",
             "Channel scheduling: how long the vessel is inside the port"],
        ],
        col_widths=cols
    ))

    # Data Source
    story.append(Paragraph("4. Data Source", styles["SectionHead"]))
    story.append(Paragraph(
        "<b>File:</b> BBDD limpia(1).xlsx, sheet \"Resume Naves Comerciales (4)\"<br/>"
        "<b>Records:</b> 5,605 vessel calls (5,597 after cleaning)<br/>"
        "<b>Period:</b> December 2019 to August 2025<br/>"
        "<b>Completeness:</b> All timestamp and numeric columns are 100% complete. "
        "Only shipping line and service route have missing values (~30%), concentrated "
        "in non-container vessel types.",
        styles["BodyText2"]
    ))

    # Feature Engineering
    story.append(Paragraph("5. Feature Engineering", styles["SectionHead"]))
    story.append(Paragraph(
        "Features use only pre-berthing information (no data leakage):",
        styles["BodyText2"]
    ))
    features = [
        ("Direct features", "vessel type, gross tonnage, terminal, agency, shipping line, "
         "arrival drafts, origin/destination ports and regions"),
        ("Temporal features", "month, day of week, hour, year, quarter, weekend flag"),
        ("Historical features", "expanding-window statistics of the same vessel's prior berth "
         "stays (mean, median, std, last visit, count), with shift-by-1 to prevent leakage"),
        ("Group-level features", "expanding-window averages at (vessel type, terminal), "
         "vessel type, and terminal levels"),
    ]
    for name, desc in features:
        story.append(Paragraph(f"<b>{name}:</b> {desc}", styles["SmallText"]))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Berth assignment (Sitio) is excluded as it is the BAP decision variable. "
        "Departure drafts are excluded as they are not known pre-berthing.",
        styles["BodyText2"]
    ))

    # Vessel Type Grouping
    story.append(Paragraph("6. Vessel Type Grouping", styles["SectionHead"]))
    story.append(Paragraph(
        "The 16 original vessel types are aggregated into 7 operationally meaningful groups "
        "based on cargo handling methods and service time patterns:",
        styles["BodyText2"]
    ))
    cols = [1.2*inch, 2.8*inch, 0.7*inch, 1.0*inch]
    story.append(make_table(
        ["Group", "Original Types", "Count", "Avg Stay (h)"],
        [
            ["Container", "Contenedor", "3,066", "36"],
            ["Dry Bulk", "Carga Seca Granel, Mineral/Granel/Petrolero", "951", "74"],
            ["Vehicle Carrier", "Autero, Autotrasbordo", "749", "36"],
            ["Liquid Bulk", "Transp. Quimico/Liquido/Asfalto, Petrolero", "587", "22"],
            ["General Cargo", "Tradicional, C. Proyecto, Chipero, Refrig., Otros", "171", "36"],
            ["Passenger", "Pasajeros", "80", "19"],
            ["Other", "Nave Armada", "1", "65"],
        ],
        col_widths=cols
    ))

    # Data Cleaning
    story.append(Paragraph("7. Data Cleaning", styles["SectionHead"]))
    cleaning = [
        "7 records with berth stay < 2 hours removed (aborted calls or data errors)",
        "1 record with berth stay of 780 hours removed (anomalous)",
        "12 records flagged where dispatch timestamp precedes reception (quality issue, targets remain valid)",
    ]
    for item in cleaning:
        story.append(Paragraph(f"\u2022 {item}", styles["BodyText2"]))

    # Train/Test Split
    story.append(Paragraph("8. Suggested Train/Test Split", styles["SectionHead"]))
    story.append(Paragraph(
        "A temporal split prevents data leakage from the historical features:",
        styles["BodyText2"]
    ))
    cols = [1.2*inch, 2.5*inch, 1.0*inch]
    story.append(make_table(
        ["Split", "Period", "Records"],
        [
            ["Train", "Before 2024-07-01", "4,413"],
            ["Validation", "2024-07-01 to 2025-02-28", "688"],
            ["Test", "After 2025-03-01", "496"],
        ],
        col_widths=cols
    ))

    # External Data Enrichment
    story.append(Paragraph("9. External Data Enrichment", styles["SectionHead"]))
    story.append(Paragraph(
        "External datasets were collected to supplement the main vessel call database. "
        "All external data is stored in the external_data/ folder with full source documentation "
        "in sources.md.",
        styles["BodyText2"]
    ))

    story.append(Paragraph("Weather Data (Open-Meteo API, CC BY 4.0)", styles["SubHead"]))
    weather_items = [
        "weather_daily.csv: 2,056 days of temperature, precipitation, wind speed/gusts/direction (complete)",
        "weather_hourly.csv: 49,344 hours of wind and precipitation data (complete)",
        "marine_weather_daily.csv: 2,056 days of wave height, period, swell (31% null before Oct 2021)",
    ]
    for item in weather_items:
        story.append(Paragraph(f"\u2022 {item}", styles["SmallText"]))
        story.append(Spacer(1, 2))
    story.append(Paragraph(
        "Weather is the most impactful enrichment: wind and wave conditions directly affect "
        "mooring operations, port entry, and can extend service times.",
        styles["BodyText2"]
    ))

    story.append(Paragraph("Vessel Characteristics", styles["SubHead"]))
    story.append(Paragraph(
        "vessel_dimensions.csv: LOA (length overall) for 338 of 1,550 vessels (22%), "
        "derived by cross-matching IMO codes through imo_vessel_codes.csv (64K vessels, GitHub, "
        "Public Domain) and vessel_information_ais.csv (10K vessels with LOA). Fuller coverage "
        "requires Kaggle account or paid databases (Datalastic, MarineTraffic).",
        styles["BodyText2"]
    ))

    story.append(Paragraph("Reference Datasets", styles["SubHead"]))
    ref_items = [
        "port_noumea_stopovers.csv: ~50K port calls from Noumea (2002-2017), 35 vessels overlap with our fleet (Zenodo)",
        "ship_movement_2015-2022.zip: Berth mooring dynamics, 14 ships with length/beam/DWT and weather (Zenodo, CC BY)",
    ]
    for item in ref_items:
        story.append(Paragraph(f"\u2022 {item}", styles["SmallText"]))
        story.append(Spacer(1, 2))

    # Known Limitations
    story.append(Paragraph("10. Known Limitations", styles["SectionHead"]))
    limitations = [
        "No cargo volume data in the main database. Container counts exist only for ~90 vessels "
        "in stats plan 2025. Draft difference could proxy for volume but is only available post-berthing.",
        "No crane/equipment data. Cannot normalize service time by handling resources.",
        "Vessel dimensions (LOA, beam, DWT) available for only 22% of vessels from free sources.",
        "30% missing shipping line/service data for non-container types. Structurally missing "
        "(bulk/tanker vessels don't use liner services).",
        "Weather data not yet joined to training dataset. Available in external_data/ for integration.",
        "Marine wave data has gaps before October 2021 (31% null).",
    ]
    for item in limitations:
        story.append(Paragraph(f"\u2022 {item}", styles["BodyText2"]))

    # Project Structure
    story.append(Paragraph("11. Project Structure", styles["SectionHead"]))
    files = [
        ["data/BBDD limpia(1).xlsx", "Source data (unchanged)"],
        ["data/training_dataset.csv", "Generated training dataset (5,597 rows x 44 cols)"],
        ["data/column_description.pdf", "Column definitions (EN/ES)"],
        ["data/project_description.pdf", "This document"],
        ["external_data/", "Weather, vessel dimensions, reference datasets + sources.md"],
        ["src/build_training_dataset.py", "Dataset builder script"],
        ["src/download_external_data.py", "External data downloader"],
        ["src/port_regions.py", "Port-to-region mapping dictionary"],
        ["src/generate_pdfs.py", "PDF documentation generator"],
    ]
    cols = [2.5*inch, 4.0*inch]
    story.append(make_table(
        ["File", "Description"],
        files,
        col_widths=cols
    ))

    doc.build(story)
    print(f"  Created {filepath}")


def main():
    print("Generating PDF documentation...")
    build_column_description_pdf()
    build_project_description_pdf()
    print("Done.")


if __name__ == "__main__":
    main()

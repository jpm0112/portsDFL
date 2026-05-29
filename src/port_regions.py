"""
Port-to-region mapping for San Antonio BAP service time prediction.

Maps 132 origin ports and 110 destination ports to geographic regions.
Used by build_training_dataset.py to create origin_region and dest_region features.
"""

PORT_TO_REGION = {
    # --- Chilean North ---
    "ARICA": "Chilean_North",
    "IQUIQUE": "Chilean_North",
    "ANTOFAGASTA": "Chilean_North",
    "MEJILLONES": "Chilean_North",
    "PATILLOS": "Chilean_North",
    "CALETA PATILLOS": "Chilean_North",
    "TOCOPILLA": "Chilean_North",
    "CALDERA": "Chilean_North",
    "CHANARAL": "Chilean_North",
    "ANGAMOS": "Chilean_North",
    "BARQUITO": "Chilean_North",
    "GUAYACAN": "Chilean_North",
    "LOS VILOS": "Chilean_North",
    "CHANCAN": "Chilean_North",
    "COQUIMBO": "Chilean_North",
    "HUASCO": "Chilean_North",

    # --- Chilean Central ---
    "VALPARAISO": "Chilean_Central",
    "SAN ANTONIO - CL": "Chilean_Central",
    "QUINTERO": "Chilean_Central",
    "SAN VICENTE": "Chilean_Central",
    "CORONEL": "Chilean_Central",
    "LIRQUEN": "Chilean_Central",
    "LAS SALINAS/VALPARAISO": "Chilean_Central",
    "PENCO": "Chilean_Central",
    "TALCAHUANO": "Chilean_Central",

    # --- Chilean South ---
    "PUERTO MONTT": "Chilean_South",
    "PUNTA ARENAS": "Chilean_South",
    "CHACABUCO-PTO AYSEN": "Chilean_South",
    "CABO NEGRO": "Chilean_South",
    "ISLA JUAN FERNANDEZ": "Chilean_South",
    "ISLA DE PASCUA": "Chilean_South",

    # --- Peru ---
    "CALLAO": "Peru",
    "ILO": "Peru",
    "MATARANI": "Peru",
    "PAITA": "Peru",
    "LIMA": "Peru",
    "PISCO-GEN SAN MARTIN": "Peru",
    "OQUENDO": "Peru",
    "CHANCAY": "Peru",

    # --- Brazil ---
    "SANTOS - SP": "Brazil",
    "RIO GRANDE": "Brazil",
    "RIO GRANDE - RS": "Brazil",
    "RIO DE JANEIRO - RJ": "Brazil",
    "RIO DE JANEIRO-GALEAO APT": "Brazil",
    "PARANAGUA - PR": "Brazil",
    "VITORIA - ES": "Brazil",
    "BRASILIA - DF": "Brazil",
    "SAN FRANCISCO DO SUL": "Brazil",
    "SAO PAULO-COGONHAS APT": "Brazil",
    "SAO PAULO-VIRACOPOS APT": "Brazil",

    # --- Argentina & Uruguay ---
    "BUENOS AIRES": "Argentina",
    "ROSARIO": "Argentina",
    "ZARATE": "Argentina",
    "BAHIA BLANCA": "Argentina",
    "NECOCHEA": "Argentina",
    "CAMPANA": "Argentina",
    "SAN LORENZO": "Argentina",
    "SAN NICOLAS": "Argentina",
    "PUERTO MADRYN": "Argentina",
    "USHUAIA": "Argentina",
    "MONTEVIDEO": "Argentina",
    "NUEVA PALMIRA": "Argentina",
    "PUNTA DEL ESTE": "Argentina",

    # --- Ecuador & Colombia ---
    "GUAYAQUIL": "Ecuador_Colombia",
    "BUENAVENTURA": "Ecuador_Colombia",
    "MANTA": "Ecuador_Colombia",
    "PUERTO BOLIVAR": "Ecuador_Colombia",
    "CARTAGENA - CO": "Ecuador_Colombia",

    # --- Central America & Caribbean ---
    "BALBOA": "Central_America",
    "CRISTOBAL": "Central_America",
    "PANAMA CITY": "Central_America",
    "MANZANILLO - MX": "Central_America",
    "MANZANILLO PANAMA": "Central_America",
    "LAZARO CARDENAS - MIC": "Central_America",
    "ACAJUTLA": "Central_America",
    "CORINTO": "Central_America",
    "PUERTO QUETZAL": "Central_America",
    "QUETZAL": "Central_America",
    "KINGSTON": "Central_America",
    "ACAPULCO - GRO": "Central_America",
    "MAZATLAN - SIN": "Central_America",
    "PUNTARENAS": "Central_America",
    "SAN JUAN": "Central_America",
    "SOUTH CAICOS": "Central_America",
    "MEXICO CITY": "Central_America",
    "KIRA KIRA - SAN CRISTOBAL ISLAND": "Central_America",

    # --- North America ---
    "HOUSTON - TX": "North_America",
    "LONG BEACH - CA": "North_America",
    "LOS ANGELES - CA": "North_America",
    "BALTIMORE - MD": "North_America",
    "PORTLAND - ME": "North_America",
    "PORTLAND - OR": "North_America",
    "GALVESTON - TX": "North_America",
    "NEW ORLEANS INT. APT": "North_America",
    "CORPUS CHRISTI": "North_America",
    "DESTREHAN - LA": "North_America",
    "GARY - IN": "North_America",
    "GEISMAR, LA / U.S.A.": "North_America",
    "KALAMA,WASHINGTON USA": "North_America",
    "LOUISIANA": "North_America",
    "RESERVE": "North_America",
    "WILMINGTON - DE": "North_America",
    "JACKSONVILLE - FL": "North_America",
    "SAN FRANCISCO - CA": "North_America",
    "COLUMBIA - WN": "North_America",
    "GRAYS RIVER - WN": "North_America",
    "FRASER MILLS - BC": "North_America",
    "PRINCE RUPERT - BC": "North_America",
    "VANCOUVER - BC": "North_America",
    "VANCOUVER APT - BC": "North_America",
    "PORTSMOUTH": "North_America",
    "MISSIMI": "North_America",
    "SIDNEY - OH": "North_America",

    # --- Asia ---
    "SHANGHAI": "Asia",
    "HONG KONG": "Asia",
    "BUSAN": "Asia",
    "PUSAN KOREA": "Asia",
    "SINGAPORE": "Asia",
    "NINGBO": "Asia",
    "TIANJIN": "Asia",
    "XINGANG": "Asia",
    "SHENZHEN": "Asia",
    "SHEKOU": "Asia",
    "GUANGZHOU": "Asia",
    "NANSHA, GUANGDONG": "Asia",
    "FANGCHENG": "Asia",
    "LIANYUNGANG": "Asia",
    "CAM PHA": "Asia",
    "YANTAI, CHINA": "Asia",
    "YANTIAN": "Asia",
    "XIAMEN": "Asia",
    "CHINA - KAGOSHIMA": "Asia",
    "LAEMCHABANG": "Asia",
    "Laem Chabang": "Asia",
    "ULSAN": "Asia",
    "KWANGYANG": "Asia",
    "MASAN": "Asia",
    "GUNSAN ( ES KUNSAN )": "Asia",
    "BUKPYUNG PORT": "Asia",
    "PYONGTAEK": "Asia",
    "INCHEON": "Asia",
    "INCHON": "Asia",
    "YOKOHAMA - KANAGAWA": "Asia",
    "TOYOHASHI - AICHI": "Asia",
    "HIROSHIMA - HIROSHIMA": "Asia",
    "TSUKUMI - OITA": "Asia",
    "UBE - YAMAGUCHI": "Asia",
    "NAGOYA - AICHI": "Asia",
    "NAGASAKI - NAGASAKI": "Asia",
    "KOBE - HYOGO": "Asia",
    "KANDA (KARITA), FUKUOKA": "Asia",
    "TOKYO - TOKYO": "Asia",
    "TAIYUAN": "Asia",
    "SRINAGAR": "Asia",

    # --- Europe ---
    "LONDON": "Europe",
    "ZEEBRUGGE": "Europe",
    "ROTTERDAM": "Europe",
    "LIVORNO": "Europe",
    "VIGO": "Europe",
    "HUELVA": "Europe",
    "BOURGAS": "Europe",

    # --- Africa ---
    "CAPETOWN": "Africa",
    "DURBAN": "Africa",
    "PORT ELIZABETH": "Africa",
    "EAST LONDON": "Africa",

    # --- Oceania ---
    "MELBOURNE-CITY HELIPORT - VI": "Oceania",
    "FREMANTLE - WA": "Oceania",
    "TAURANGA": "Oceania",
    "SYDNEY - NS": "Oceania",
    "PAPEETE": "Oceania",
}


def get_region(port_name):
    """
    Returns the geographic region for a port name.
    Falls back to 'Other' for unmapped ports.

    Input: port_name (str)
    Output: region string
    """
    if not port_name or str(port_name).strip() == "":
        return "Other"
    return PORT_TO_REGION.get(str(port_name).strip(), "Other")

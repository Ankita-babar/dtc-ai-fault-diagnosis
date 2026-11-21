"""
Domain rules used to (a) structure-parse any DTC and (b) create 'silver' labels for codes that
have no expert label.  Gold labels (data/shortlisted_dtc_project.csv) are ONLY used as the test set
and to learn the group->severity prior; they are never edited by these rules.
"""
import re
import pandas as pd

# ---- SAE J2012 structure: P0 3 01 -> letter | 0=generic,1=manufacturer,... | subsystem digit | 2 digits
P_SUBSYSTEM = {"0": "Air intake & fuel metering", "1": "Air intake & fuel metering",
               "2": "Fuel delivery & injection", "3": "Ignition & misfire", "4": "Emission control",
               "5": "Idle, speed & oil pressure", "6": "ECU / computer", "7": "Transmission",
               "8": "Transmission", "9": "Transmission", "A": "Charging & electrical",
               "B": "Charging & electrical", "C": "Charging & electrical"}
LETTER = {"P": "Powertrain", "B": "Body", "C": "Chassis", "U": "Network"}


def parse_code(code: str) -> dict:
    c = str(code).strip().upper()
    ok = bool(re.fullmatch(r"[PBCU][0-9A-F]{4}", c))
    return {"valid": ok, "letter": c[:1], "p1": c[:2], "p2": c[:3],
            "generic": ok and c[1] in "02" or (ok and c[:3] in ("P34", "P35", "P36", "P37", "P38", "P39")),
            "category": LETTER.get(c[:1], "Unknown")}


def range_group(code: str):
    """Best guess of the sub-system from the code pattern alone (works with NO description)."""
    c = str(code).upper()
    if c[:1] == "P" and len(c) == 5:
        if c[1] == "3":                       # P3xxx: cylinder deactivation / valve control / manufacturer-specific
            return "Air intake & fuel metering"
        return P_SUBSYSTEM.get(c[2])
    if c[:1] == "B": return "Airbag (SRS) deployment"
    if c[:1] == "C": return "ABS / wheel speed"
    if c[:1] == "U": return "Lost communication"
    return None


# ---- keyword -> maintenance group (first match wins) --------------------------------------------
GROUP_RULES = [
    (r"misfire|ignition coil|spark", "Ignition & misfire"),
    (r"knock|crankshaft|camshaft|crank/cam", "Crank/cam & knock sensors"),
    (r"valve control|deactivation|valve lift|valve timing", "Air intake & fuel metering"),
    (r"turbo|supercharger|boost(?!er)|glow plug|diesel|particulate|exhaust pressure", "Turbo & diesel systems"),
    (r"oxygen|\bo2\b|\bho2s\b|air/fuel ratio|lambda|a/f sensor", "Oxygen sensors"),
    (r"fuel trim|system too (lean|rich)", "Air intake & fuel metering"),
    (r"evap|purge|vent control|gas cap|fuel cap|secondary air|\bair system\b|catalyst|\begr\b|exhaust gas recirc|nox|reductant|scr\b", "Emission control"),
    (r"injector|fuel pump|fuel rail|fuel pressure|fuel volume|fuel delivery|fuel system pressure|fuel temperature|fuel level", "Fuel delivery & injection"),
    (r"seat belt|pretension|buckle", "Seat belt"),
    (r"roll ?over|restraints? sensor|crash sensor|impact sensor", "Crash / restraint sensors"),
    (r"airbag|deployment|restraint|occupant", "Airbag (SRS) deployment"),
    (r"tire|tyre", "Tyre pressure"),
    (r"wheel speed|\babs\b|anti-lock|traction", "ABS / wheel speed"),
    (r"brake|booster", "Brake system"),
    (r"steering|yaw|lateral|stability", "Steering & stability"),
    (r"throttle|pedal|accelerator", "Throttle-by-wire & pedal"),
    (r"transmission|transaxle|gear|clutch|torque converter|shift solenoid|\btcc\b|pressure control solenoid|turbine|input shaft|output shaft", "Transmission"),
    (r"coolant|thermostat|cooling fan|fan \d|radiator|engine over ?temp|engine temperature", "Engine cooling & thermal"),
    (r"mass or volume air|maf\b|manifold|intake air|barometric|air flow|intake manifold", "Air intake & fuel metering"),
    (r"oil pressure|oil level|idle|engine speed|vehicle speed|cruise|rpm|speed sensor", "Idle, speed & oil pressure"),
    (r"battery|charging|generator|alternator|system voltage|power supply|ignition switch|ignition (on|off) |starter|hybrid|drive motor", "Charging & electrical"),
    (r"software|calibration|configuration|not programmed|programming|incompatible", "Software"),
    (r"lost communication", "Lost communication"),
    (r"invalid data|invalid serial|implausible", "Invalid data"),
    (r"bus off|\bcan\b|\bbus\b|network", "CAN bus"),
    (r"control module|\becm\b|\bpcm\b|processor|memory|checksum|\bram\b|\brom\b|eeprom|serial communication|watchdog", "ECU / computer"),
    (r"communication", "CAN bus"),
]
GROUP_RE = [(re.compile(p, re.I), g) for p, g in GROUP_RULES]

# ---- severity overrides: general automotive knowledge (fixed, not tuned on the test labels) ------
HIGH_RE = re.compile(r"misfire|over ?temperature|overheat|rail/system pressure.*(low|high)|oil pressure too low|fuel pump|"
                     r"injector.*circuit|crankshaft|camshaft position|ignition coil|deployment|roll ?over|pretension|wheel speed|"
                     r"brake fluid|brake booster|processor|checksum|\bram\b|forced limited power|incorrect gear ratio|"
                     r"bus off|can .*bus|overboost|underboost|lost communication with (ecm|pcm|tcm|abs|brake|restraints|gateway)", re.I)
LOW_RE = re.compile(r"evap|purge|vent control|gas cap|fuel cap|tire|tyre|thermostat|mil control|insufficient coolant temp|"
                    r"closed loop|lining wear|small leak|air system|intake air temperature|secondary air", re.I)


U_RULES = [(re.compile(p, re.I), g) for p, g in [
    (r"lost communication", "Lost communication"), (r"invalid data|implausible", "Invalid data"),
    (r"software|calibration|configuration|not programmed|incompatible", "Software"),
    (r"bus off|\bcan\b|\bbus\b|network|communication", "CAN bus")]]


def rule_group(code, desc):
    d = str(desc or "")
    if str(code).upper().startswith("U"):
        for rx, g in U_RULES:
            if rx.search(d):
                return g
    for rx, g in GROUP_RE:
        if rx.search(d):
            return g
    return range_group(code) or "Other"


def rule_severity(desc, group, prior: dict):
    d = str(desc or "")
    if LOW_RE.search(d): return "Low"
    if HIGH_RE.search(d): return "High"
    return prior.get(group, "Medium")


def group_prior(gold: pd.DataFrame) -> dict:
    """Most common severity of each maintenance group in the (training) gold labels."""
    return gold.groupby("maintenance_group")["severity"].agg(lambda s: s.value_counts().idxmax()).to_dict()

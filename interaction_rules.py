"""Small portfolio-demo interaction rule set.

This is deliberately labelled as incomplete and must not be treated as a
clinical interaction database.
"""


INTERACTION_RULES = {
    ("aspirin", "ibuprofen"): (
        "danger",
        "Ibuprofen may antagonize the antiplatelet effect of low-dose aspirin. Combined use increases GI bleeding risk.",
    ),
    ("aspirin", "warfarin"): (
        "danger",
        "Significantly increased bleeding risk. Professional verification is required.",
    ),
    ("warfarin", "amoxicillin"): (
        "caution",
        "Broad-spectrum antibiotics may alter anticoagulation. Verify and monitor through a qualified professional.",
    ),
    ("atorvastatin", "warfarin"): (
        "caution",
        "A possible interaction is represented in this demo rule set. Professional verification is required.",
    ),
    ("aspirin", "clopidogrel"): (
        "danger",
        "Combined antiplatelet therapy can increase bleeding risk. Professional verification is required.",
    ),
    ("ibuprofen", "warfarin"): (
        "danger",
        "This combination can increase bleeding risk. Professional verification is required.",
    ),
    ("omeprazole", "clopidogrel"): (
        "caution",
        "A possible CYP2C19-related interaction is represented in this demo rule set. Professional verification is required.",
    ),
    ("lisinopril", "aspirin"): (
        "caution",
        "A possible blood-pressure or renal interaction is represented in this demo rule set. Professional verification is required.",
    ),
    ("阿司匹林", "布洛芬"): (
        "danger",
        "演示规则提示可能增加胃肠道出血风险，必须由专业人员核实。",
    ),
    ("阿司匹林", "华法林"): (
        "danger",
        "演示规则提示可能增加出血风险，必须由专业人员核实。",
    ),
    ("华法林", "头孢克肟"): (
        "caution",
        "演示规则提示可能影响抗凝效果，请由专业人员核实。",
    ),
    ("辛伐他汀", "氯吡格雷"): (
        "caution",
        "演示规则中记录了潜在相互作用，请由专业人员核实。",
    ),
    ("阿司匹林", "氯吡格雷"): (
        "danger",
        "演示规则提示联合使用可能增加出血风险，必须由专业人员核实。",
    ),
    ("布洛芬", "华法林"): (
        "danger",
        "演示规则提示可能增加胃肠道出血风险，必须由专业人员核实。",
    ),
    ("奥美拉唑", "氯吡格雷"): (
        "caution",
        "演示规则中记录了潜在相互作用，请由专业人员核实。",
    ),
}


COMMON_DRUGS_EN = [
    "Aspirin", "Ibuprofen", "Acetaminophen", "Warfarin", "Amoxicillin",
    "Omeprazole", "Metformin", "Atorvastatin", "Lisinopril", "Metoprolol",
    "Clopidogrel", "Simvastatin", "Methotrexate",
]
COMMON_DRUGS_ZH = [
    "阿司匹林", "布洛芬", "华法林", "头孢克肟", "奥美拉唑",
    "二甲双胍", "氨氯地平", "氯吡格雷", "辛伐他汀", "甲氨蝶呤",
]


def get_drug_list(language):
    return COMMON_DRUGS_EN if language == "en" else COMMON_DRUGS_ZH


def find_interaction(drug_a, drug_b):
    a, b = drug_a.lower(), drug_b.lower()
    return INTERACTION_RULES.get((a, b)) or INTERACTION_RULES.get((b, a))


"""Human-reviewed WATI labels for visual identity eval (issue #3)."""

from __future__ import annotations

# Confirmed exact SKU. Events 726/722 are 53566_003 (customer photo is that model).
# 721/727 vs 49452_B is a permanent hard-negative (stone/hardware sister models).
CONFIRMED: dict[int, str] = {
    887: "53771_003",
    845: "52792_006",
    576: "52346_B",
    863: "53698_003",
    862: "53698_003",
    861: "53698_003",
    840: "51604_A",
    748: "52010_006",
    727: "54101_003",
    721: "54101_003",
    726: "53566_003",
    722: "53566_003",
    660: "53603_004",
    885: "51591_017",
    864: "53698_003",
    1070: "53235_007",
    1072: "51604_A",
    1068: "51604_A",
    1071: "52207_A",
}

NOT_SHOES: set[int] = {736}
ORDER_SCREENSHOTS: set[int] = {753, 1162}
DELIVERY_NOTICES: set[int] = {1057}
UNCERTAIN_LABELS: set[int] = {865}
UNLABELED: set[int] = {869, 880, 611, 589, 568, 559}

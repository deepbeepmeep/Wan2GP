from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scripts.dual_run_compare.thresholds import APPROVED_CALIBRATION_STATUSES


ReportStatus = Literal["green", "red", "pending", "fallback", "wgp_only"]
SectionStatus = Literal[
    "green",
    "red",
    "pending",
    "missing_evidence",
    "pending_not_implemented",
    "not_applicable",
    "wgp_only",
]

GREEN: ReportStatus = "green"
RED: ReportStatus = "red"
PENDING: ReportStatus = "pending"
FALLBACK: ReportStatus = "fallback"
WGP_ONLY: ReportStatus = "wgp_only"

SECTION_GREEN: SectionStatus = "green"
SECTION_RED: SectionStatus = "red"
SECTION_PENDING: SectionStatus = "pending"
SECTION_MISSING_EVIDENCE: SectionStatus = "missing_evidence"
SECTION_PENDING_NOT_IMPLEMENTED: SectionStatus = "pending_not_implemented"
SECTION_NOT_APPLICABLE: SectionStatus = "not_applicable"
SECTION_WGP_ONLY: SectionStatus = "wgp_only"

REPORT_STATUSES: frozenset[str] = frozenset({"green", "red", "pending", "fallback", "wgp_only"})
SECTION_STATUSES: frozenset[str] = frozenset(
    {
        "green",
        "red",
        "pending",
        "missing_evidence",
        "pending_not_implemented",
        "not_applicable",
        "wgp_only",
    }
)


CALIBRATION_TO_INITIAL_REPORT_STATUS: dict[str, ReportStatus] = {
    "green": PENDING,
    "pending_calibration": PENDING,
    "deferred_pending_sprint_0c_disk": PENDING,
    "owner_deferred": PENDING,
    "wgp_only": WGP_ONLY,
}


@dataclass(frozen=True)
class SectionStatusRecord:
    key: str
    status: SectionStatus
    reason: str | None = None

    def as_dict(self) -> dict[str, str]:
        payload = {"key": self.key, "status": self.status}
        if self.reason:
            payload["reason"] = self.reason
        return payload


def map_calibration_status_to_report_status(calibration_status: str) -> ReportStatus:
    if calibration_status not in APPROVED_CALIBRATION_STATUSES:
        raise ValueError(f"unknown calibration status: {calibration_status}")
    return CALIBRATION_TO_INITIAL_REPORT_STATUS[calibration_status]


def pending_not_implemented_section(key: str, reason: str) -> dict[str, str]:
    return SectionStatusRecord(
        key=key,
        status=SECTION_PENDING_NOT_IMPLEMENTED,
        reason=reason,
    ).as_dict()

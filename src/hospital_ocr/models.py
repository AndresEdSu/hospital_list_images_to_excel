from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ImageSource:
    path: Path
    center_slug: str
    center_name: str


@dataclass(frozen=True)
class OcrLine:
    text: str
    score: float
    box: tuple[int, int, int, int]
    image_width: int
    image_height: int

    @property
    def center_x(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def center_y(self) -> float:
        return (self.box[1] + self.box[3]) / 2


@dataclass(frozen=True)
class Specialty:
    alias: str
    specialty: str
    area: str = ""


@dataclass(frozen=True)
class Place:
    alias: str
    name: str


@dataclass
class PatientRecord:
    full_name: str
    first_name: str
    last_name: str
    name_split_confidence: float
    detected_name_order: str
    center: str
    age: int | None
    age_unit: str
    sex: str
    origin: str
    specialty: str
    area: str
    source_image: str
    confidence: float
    needs_review: bool
    notes: list[str] = field(default_factory=list)
    raw_line: str = ""
    occurrences: int = 1
    patient_id: str = ""
    source_images: list[str] = field(default_factory=list)
    duplicate_status: str = "Único"
    duplicate_detail: str = ""
    document_id: str = ""
    review_status: str = ""
    clinical_notes: str = ""
    name_confidence: float = 0.0
    document_confidence: float = 0.0
    age_confidence: float = 0.0
    origin_confidence: float = 0.0
    specialty_confidence: float = 0.0
    field_evidence: dict[str, str] = field(default_factory=dict)

    def add_note(self, note: str) -> None:
        if note and note not in self.notes:
            self.notes.append(note)

    @property
    def notes_text(self) -> str:
        return "; ".join(self.notes)

    @property
    def observations_text(self) -> str:
        values = [self.clinical_notes, *self.notes]
        return "; ".join(dict.fromkeys(value for value in values if value))

    @property
    def source_images_text(self) -> str:
        images = self.source_images or [self.source_image]
        return "; ".join(dict.fromkeys(images))

    def add_duplicate_detail(self, detail: str) -> None:
        details = self.duplicate_detail.split("; ") if self.duplicate_detail else []
        if detail and detail not in details:
            details.append(detail)
        self.duplicate_detail = "; ".join(details)

    @property
    def field_evidence_text(self) -> str:
        return "; ".join(
            f"{field}: {evidence}"
            for field, evidence in self.field_evidence.items()
            if evidence
        )


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    patient_id: str
    record: PatientRecord


@dataclass(frozen=True)
class ConsolidationResult:
    patients: list[PatientRecord]
    evidence: list[EvidenceRecord]

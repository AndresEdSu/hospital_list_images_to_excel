"""Internal components for extracting structured hospital tables."""

from hospital_ocr.table_extraction.parser import (
    looks_like_table,
    parse_table_lines,
)

__all__ = ["looks_like_table", "parse_table_lines"]

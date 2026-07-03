"""Backward-compatible facade for table extraction."""

from hospital_ocr.table_extraction import looks_like_table, parse_table_lines

__all__ = ["looks_like_table", "parse_table_lines"]

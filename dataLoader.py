# dataLoader.py
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

import pandas as pd


def _safe_get_first_dns_value(dns_obj: Dict[str, Any], key: str) -> Any:
    """Extracts values such as dns.rrname from Suricata DNS query/answer lists."""
    for list_key in ("queries", "answers", "grouped"):
        value = dns_obj.get(list_key)
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, dict) and key in first:
                return first.get(key)
    return dns_obj.get(key)


def _normalize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes nested Suricata EVE structures before pandas.json_normalize."""
    dns = event.get("dns")
    if isinstance(dns, dict):
        if "rrname" not in dns:
            rrname = _safe_get_first_dns_value(dns, "rrname")
            if rrname is not None:
                dns["rrname"] = rrname
        if "rcode" not in dns:
            rcode = _safe_get_first_dns_value(dns, "rcode")
            if rcode is not None:
                dns["rcode"] = rcode
    return event


def _fallback_flow_id(event: Dict[str, Any], line_number: int) -> str:
    """Creates a deterministic fallback id if Suricata flow_id is missing."""
    fields = [
        str(event.get("timestamp", "")),
        str(event.get("src_ip", "")),
        str(event.get("src_port", "")),
        str(event.get("dest_ip", "")),
        str(event.get("dest_port", "")),
        str(event.get("proto", "")),
        str(line_number),
    ]
    raw = "|".join(fields).encode("utf-8", errors="ignore")
    return "generated_" + hashlib.sha1(raw).hexdigest()[:16]


def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads Suricata EVE JSON-lines and returns two aligned DataFrames:
      1. ml_ready_df: raw flattened rows before feature engineering
      2. human_readable_df: same rows, kept for explanations and validation

    Rows are grouped by flow_id using first non-null values per column. The flow_id is
    kept as the DataFrame index and also as a normal column for export/debugging.
    """
    parsed_json_list: List[Dict[str, Any]] = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event = _normalize_event(event)
            if "flow_id" not in event or pd.isna(event.get("flow_id")):
                event["flow_id"] = _fallback_flow_id(event, line_number)
            parsed_json_list.append(event)

    if not parsed_json_list:
        raise ValueError(f"No valid JSON events were loaded from: {file_path}")

    df = pd.json_normalize(parsed_json_list)

    if "flow_id" not in df.columns:
        raise ValueError("The loaded data has no flow_id column and fallback generation failed.")

    # groupby.first() returns first non-null value per column, which helps combine flow/alert fields.
    df = df.groupby("flow_id", sort=False).first()
    df["flow_id"] = df.index.astype(str)

    human_readable_df = df.copy()
    ml_ready_df = df.copy()

    return ml_ready_df, human_readable_df

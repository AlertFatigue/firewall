# classifier.py
from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

import config

try:
    import google.generativeai as genai
except ImportError:  # Allows offline testing without Gemini package installed.
    genai = None


def row_to_llm_context(row: pd.Series) -> str:
    """Translates a human-readable Suricata row into concise SOC analyst context."""
    src = row.get("src_ip", "Unknown")
    dst = row.get("dest_ip", "Unknown")
    sport = row.get("src_port", "Unknown")
    dport = row.get("dest_port", "Unknown")

    context = [
        "Network Flow Analysis:",
        f"- Connection: {src}:{sport} -> {dst}:{dport}",
        f"- Protocol: {row.get('proto', 'Unknown')} | Application Layer: {row.get('app_proto', 'None')}",
        (
            f"- Flow Dynamics: Sent {row.get('flow.pkts_toserver', 0)} packets "
            f"({row.get('flow.bytes_toserver', 0)} bytes). Received "
            f"{row.get('flow.pkts_toclient', 0)} packets "
            f"({row.get('flow.bytes_toclient', 0)} bytes)."
        ),
        f"- Connection State: {row.get('flow.state', 'Unknown')} (Reason: {row.get('flow.reason', 'Unknown')})",
    ]

    if pd.notna(row.get("http.hostname")):
        context.append(
            f"- HTTP Metadata: Hostname '{row.get('http.hostname')}', "
            f"Method '{row.get('http.http_method', 'Unknown')}', "
            f"URL '{row.get('http.url', 'Unknown')}'"
        )
    if pd.notna(row.get("http.http_user_agent")):
        context.append(f"- HTTP User-Agent: {row.get('http.http_user_agent')}")

    if pd.notna(row.get("dns.rrname")):
        context.append(
            f"- DNS Query: Requested '{row.get('dns.rrname')}' "
            f"(Response Code: {row.get('dns.rcode', 'Unknown')})"
        )

    if pd.notna(row.get("tls.sni")):
        context.append(f"- TLS SNI: {row.get('tls.sni')}")
    if pd.notna(row.get("tls.subject")):
        context.append(f"- TLS Certificate Subject: {row.get('tls.subject')}")

    if pd.notna(row.get("alert.signature")):
        context.append(
            f"- Suricata Baseline Detection: '{row.get('alert.signature')}' "
            f"(Severity: {row.get('alert.severity', 'Unknown')}, "
            f"Category: {row.get('alert.category', 'Unknown')})"
        )

    if pd.notna(row.get("cluster_label")):
        context.append(
            f"- Cluster: {row.get('cluster_label')} "
            f"(membership probability {row.get('cluster_probability', 'Unknown')})"
        )

    return "\n".join(context)


def init_llm(threat_labels: List[str]):
    """
    Initializes Gemini. If GEMINI_API_KEY or the library is missing, returns None.
    The pipeline can still run with rule-based fallback labels for debugging.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None

    genai.configure(api_key=api_key)
    labels_string = "\n".join([f"- {label}" for label in threat_labels])

    system_instruction = f"""
[Capacity and Role]: You are an expert Cybersecurity SOC Analyst specializing in Network Intrusion Detection Systems (NIDS).
[Insight]: I am building a machine learning pipeline that uses unsupervised clustering to group network flows. You are the zero-shot classifier for cluster exemplars and outliers.
[Statement]: Analyze the provided network flow context and classify it into exactly ONE of the following Threat Categories. You must choose from this exact list:
{labels_string}
[Personality]: Objective, analytical, and strictly compliant with formatting rules.
[Experiment]: Return ONLY a valid JSON object with a single key "label" containing the exact string of the chosen category. Do not include explanations.
"""

    return genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        system_instruction=system_instruction,
        generation_config={"response_mime_type": "application/json"},
    )


def fallback_label_from_context(text: str, threat_labels: List[str]) -> str:
    """
    Deterministic fallback used only when Gemini is not available.
    This is not a replacement for the thesis LLM step; it keeps the pipeline testable.
    """
    lower = text.lower()

    if any(word in lower for word in ["heartbleed", "exploit", "attack", "trojan", "malware", "cve", "shellcode"]):
        return "Known Vulnerability Exploitation or Active Attack"
    if any(word in lower for word in ["beacon", "command and control", "c2"]):
        return "Suspicious Command and Control (C2) Beaconing"
    if "dns query" in lower or "application layer: dns" in lower:
        return "Standard DNS Resolution and Naming Services"
    if "application layer: http" in lower or "http metadata" in lower:
        return "Routine Unencrypted Web Traffic (HTTP)"
    if "application layer: tls" in lower or "tls sni" in lower or "https" in lower:
        return "Routine Encrypted Web Traffic (HTTPS/TLS)"
    if any(word in lower for word in ["icmp", "arp", "dhcp"]):
        return "Benign Background Network Noise (ICMP, ARP, DHCP)"
    if any(word in lower for word in ["smb", "ldap", "active directory"]):
        return "Standard Internal IT and Directory Services (LDAP, SMB, Active Directory)"

    return "Malformed protocol anomaly or unknown suspicious behavior"


def _parse_llm_response(response_text: str, threat_labels: List[str]) -> str:
    try:
        result_data = json.loads(response_text)
        chosen_label = result_data.get("label")
    except json.JSONDecodeError:
        chosen_label = None

    if chosen_label not in threat_labels:
        return "Malformed protocol anomaly or unknown suspicious behavior"
    return chosen_label


def get_labels(
    contexts: Iterable[str],
    model,
    threat_labels: List[str],
    sleep_seconds: Optional[float] = None,
) -> Dict[int, Tuple[str, float]]:
    """
    Labels each context. Returns {position: (label, confidence)}.

    Fix: the function now works both with Gemini and without Gemini. When Gemini is
    unavailable, confidence is 0.0 to clearly show these are fallback labels.
    """
    contexts = list(contexts)
    results_dict: Dict[int, Tuple[str, float]] = {}
    sleep_seconds = config.LLM_SLEEP_SECONDS if sleep_seconds is None else sleep_seconds

    for i, text in enumerate(contexts):
        if model is None:
            results_dict[i] = (fallback_label_from_context(text, threat_labels), 0.0)
            continue

        print(f" -> Querying Gemini API for item {i + 1} of {len(contexts)}...")
        try:
            response = model.generate_content(text)
            chosen_label = _parse_llm_response(response.text, threat_labels)
            results_dict[i] = (chosen_label, 1.0)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        except Exception as exc:  # Keep batch running if one API call fails.
            print(f"API Error on item {i}: {exc}")
            results_dict[i] = (fallback_label_from_context(text, threat_labels), 0.0)

    return results_dict

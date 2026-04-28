# extract.py
from __future__ import annotations

import argparse
import json
import os
from typing import Optional


def _norm_port(value) -> Optional[str]:
    if value is None or value == "":
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _matches_attack(event, args) -> bool:
    """
    Flexible attack filtering.

    You can match attacks by:
      - Suricata alert signature substring: --alert-contains Heartbleed
      - exact IP/port values: --attack-src-ip ... --attack-dest-ip ... --attack-dest-port ...

    This fixes the old hardcoded Heartbleed-only extraction.
    """
    if args.alert_contains:
        alert = event.get("alert", {}) if isinstance(event.get("alert"), dict) else {}
        signature = str(alert.get("signature", "")).lower()
        category = str(alert.get("category", "")).lower()
        needle = args.alert_contains.lower()
        if needle in signature or needle in category:
            return True

    checks = []
    if args.attack_src_ip:
        checks.append(str(event.get("src_ip", "")).strip() == args.attack_src_ip.strip())
    if args.attack_dest_ip:
        checks.append(str(event.get("dest_ip", "")).strip() == args.attack_dest_ip.strip())
    if args.attack_dest_port:
        checks.append(_norm_port(event.get("dest_port")) == _norm_port(args.attack_dest_port))

    return bool(checks) and all(checks)


def extract_data(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Could not find input file: {args.input}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    normal_count = 0
    attack_count = 0
    saved_count = 0

    print(f"Filtering {args.input} -> {args.output}")
    with open(args.input, "r", encoding="utf-8", errors="replace") as infile, open(
        args.output, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            is_attack = _matches_attack(event, args)
            is_normal = False

            if not is_attack and normal_count < args.max_normal_events:
                if args.normal_event_type == "any" or event.get("event_type") == args.normal_event_type:
                    is_normal = True
                    normal_count += 1

            if is_attack or is_normal:
                outfile.write(line)
                saved_count += 1
                if is_attack:
                    attack_count += 1

    print("Extraction complete.")
    print(f"Saved total events: {saved_count}")
    print(f"Saved attack events: {attack_count}")
    print(f"Saved normal/background events: {normal_count}")


def main():
    parser = argparse.ArgumentParser(description="Extract a memory-safe sample from Suricata EVE logs.")
    parser.add_argument("--input", default="./trueHeartbleedLogs/eve.json", help="Input EVE JSON-lines file.")
    parser.add_argument("--output", default="./filtered_eve.json", help="Output filtered EVE JSON-lines file.")
    parser.add_argument("--max-normal-events", type=int, default=15000)
    parser.add_argument("--normal-event-type", default="flow", help="Use 'any' or a Suricata event_type such as flow.")
    parser.add_argument("--alert-contains", default="Heartbleed", help="Attack signature/category substring.")
    parser.add_argument("--attack-src-ip", default=None)
    parser.add_argument("--attack-dest-ip", default=None)
    parser.add_argument("--attack-dest-port", default=None)
    args = parser.parse_args()
    extract_data(args)


if __name__ == "__main__":
    main()

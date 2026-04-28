# classifier.py
import pandas as pd
import google.generativeai as genai
import os
import json
import time

def row_to_llm_context(row):
    """
    Translates a raw pandas row (from human_readable_df) into a verbose, 
    highly detailed text context for the LLM.
    """
    
    # 1. Base Flow Details (Always present)
    context = [
        "Network Flow Analysis:",
        f"- Connection: Source IP {row.get('src_ip', 'Unknown')} -> Destination IP {row.get('dest_ip', 'Unknown')} (Port {row.get('dest_port', 'Unknown')})",
        f"- Protocol: {row.get('proto', 'Unknown')} | Application Layer: {row.get('app_proto', 'None')}",
        f"- Flow Dynamics: Sent {row.get('flow.pkts_toserver', 0)} packets ({row.get('flow.bytes_toserver', 0)} bytes). "
        f"Received {row.get('flow.pkts_toclient', 0)} packets ({row.get('flow.bytes_toclient', 0)} bytes).",
        f"- Connection State: {row.get('flow.state', 'Unknown')} (Reason: {row.get('flow.reason', 'Unknown')})"
    ]

    # 2. Dynamic HTTP Context
    if pd.notna(row.get('http.hostname')):
        context.append(f"- HTTP Metadata: Hostname '{row['http.hostname']}', "
                       f"Method '{row.get('http.http_method', 'Unknown')}', "
                       f"URL '{row.get('http.url', 'Unknown')}'")
    if pd.notna(row.get('http.http_user_agent')):
        context.append(f"- HTTP User-Agent: {row['http.http_user_agent']}")

    # 3. Dynamic DNS Context
    if pd.notna(row.get('dns.rrname')):
        context.append(f"- DNS Query: Requested '{row['dns.rrname']}' "
                       f"(Response Code: {row.get('dns.rcode', 'Unknown')})")

    # 4. Dynamic TLS/SSL Context
    if pd.notna(row.get('tls.sni')):
        context.append(f"- TLS SNI (Server Name Indication): {row['tls.sni']}")
    if pd.notna(row.get('tls.subject')):
        context.append(f"- TLS Certificate Subject: {row['tls.subject']}")

    # 5. Baseline Suricata Context (Crucial for answering RQ3)
    if pd.notna(row.get('alert.signature')):
        context.append(f"- Suricata Baseline Detection: '{row['alert.signature']}' "
                       f"(Severity: {row.get('alert.severity', 'Unknown')}, Category: {row.get('alert.category', 'Unknown')})")

    # Join it all together into a clean, verbose paragraph
    return "\n".join(context)

def init_llm(threat_labels):
    """
    Initializes the Gemini 2.5 Flash model with a strict CRISPE system prompt 
    and forces JSON output.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not found.")
    
    genai.configure(api_key=api_key)
    
    # Format the labels into a bulleted string for the prompt
    labels_string = "\n".join([f"- {label}" for label in threat_labels])

    # CRISPE Framework Prompt
    system_instruction = f"""
    [Capacity and Role]: You are an expert Cybersecurity SOC Analyst specializing in Network Intrusion Detection Systems (NIDS).
    [Insight]: I am building a machine learning pipeline that uses unsupervised clustering to group network flows. I need you to act as the zero-shot classifier for the cluster exemplars and outliers.
    [Statement]: Analyze the provided network flow context and classify it into exactly ONE of the following Threat Categories. You must choose from this exact list:
    {labels_string}
    
    [Personality]: Objective, analytical, and strictly compliant with formatting rules.
    [Experiment]: Return ONLY a valid JSON object with a single key "label" containing the exact string of the chosen category. Do not include explanations.
    """

    # 2.5 Flash for speed, and force JSON output
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction=system_instruction,
        generation_config={"response_mime_type": "application/json"}
    )
    
    return model

def get_labels(contexts, model, threat_labels):
    """
    Iterates through the contexts, makes the API call, and parses the JSON.
    """
    results_dict = {}
    total = len(contexts)
    for i, text in enumerate(contexts):
        print(f" -> Querying Gemini API for flow {i + 1} of {total}...")
        try:
            # Send the verbose context to Gemini
            response = model.generate_content(text)
            
            # Parse the guaranteed JSON output
            result_data = json.loads(response.text)
            chosen_label = result_data.get("label")
            
            # Fallback if the LLM hallucinated a label not on the list
            if chosen_label not in threat_labels:
                chosen_label = "Malformed protocol anomaly or unknown suspicious behavior"
                
            results_dict[i] = (chosen_label, 1.0) # Using 1.0 as a mock confidence score
            
            # Optional: A tiny sleep to ensure you don't hit free-tier rate limits
            time.sleep(3) 
            
        except Exception as e:
            print(f"API Error on flow {i}: {e}")
            results_dict[i] = ("Malformed protocol anomaly or unknown suspicious behavior", 0.0)
            
    return results_dict
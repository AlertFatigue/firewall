import pandas as pd
import google.generativeai as genai
import os
import time
import pickle
from google.api_core.exceptions import ResourceExhausted

def row_to_llm_context(row):
    """Translates a raw pandas row into a verbose text context."""
    context = [
        "Network Flow Analysis:",
        f"- Connection: Source IP {row.get('src_ip', 'Unknown')} -> Destination IP {row.get('dest_ip', 'Unknown')} (Port {row.get('dest_port', 'Unknown')})",
        f"- Protocol: {row.get('proto', 'Unknown')} | Application Layer: {row.get('app_proto', 'None')}",
        f"- Flow Dynamics: Sent {row.get('flow.pkts_toserver', 0)} packets ({row.get('flow.bytes_toserver', 0)} bytes). "
        f"Received {row.get('flow.pkts_toclient', 0)} packets ({row.get('flow.bytes_toclient', 0)} bytes)."
    ]

    if pd.notna(row.get('http.hostname')):
        context.append(f"- HTTP Metadata: Hostname '{row['http.hostname']}', "
                       f"Method '{row.get('http.http_method', 'Unknown')}', "
                       f"URL '{row.get('http.url', 'Unknown')}'")
    if pd.notna(row.get('http.http_user_agent')):
        context.append(f"- HTTP User-Agent: {row['http.http_user_agent']}")

    if pd.notna(row.get('dns.rrname')):
        context.append(f"- DNS Query: Requested '{row['dns.rrname']}' "
                       f"(Response Code: {row.get('dns.rcode', 'Unknown')})")

    if pd.notna(row.get('tls.sni')):
        context.append(f"- TLS SNI (Server Name Indication): {row['tls.sni']}")
        
    if pd.notna(row.get('alert.signature')):
        context.append(f"- Suricata Baseline Detection: '{row['alert.signature']}' "
                       f"(Severity: {row.get('alert.severity', 'Unknown')}, Category: {row.get('alert.category', 'Unknown')})")

    return "\n".join(context)

def init_llm(threat_labels):
    """Initializes the Gemini model with a strict CRISPE system prompt."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not found.")
    
    genai.configure(api_key=api_key)
    labels_string = "\n".join([f"- {label}" for label in threat_labels])

    system_instruction = f"""
    [Capacity and Role]: You are an expert Cybersecurity SOC Analyst specializing in Network Intrusion Detection Systems (NIDS).
    [Insight]: I am building a machine learning pipeline that uses unsupervised clustering to group network flows. I need you to act as the zero-shot classifier.
    [Statement]: Analyze the provided network flow context and classify it into exactly ONE of the following Threat Categories. You must choose from this exact list:
    {labels_string}
    
    [Personality]: Objective, analytical, and strictly compliant with formatting rules.
    [Experiment]: Return ONLY a valid JSON object with a single key "label" containing the exact string of the chosen category. Do not include explanations.
    """

    # Model successfully updated to Gemini 2.5 Flash
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction=system_instruction,
        generation_config={"response_mime_type": "application/json"}
    )
    
    return model

def get_labels(contexts, model, threat_labels, checkpoint_file="llm_checkpoint.pkl"):
    """
    Iterates through contexts, calls the API, and uses ATOMIC PICKLE SAVING 
    to guarantee zero data loss during a crash.
    """
    results_dict = {}
    
    # --- 1. PICKLE CHECKPOINT LOADING ---
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            saved_data = pickle.load(f)
            
            # Catch old checkpoints just in case
            results_dict = {
                int(k): tuple(v) if len(v) == 3 else (v[0], v[1], 0.0) 
                for k, v in saved_data.items()
            }
            print(f"[*] Resuming from {checkpoint_file}: {len(results_dict)} labels already processed.")

    total = len(contexts)
    for i, text in enumerate(contexts):
        if i in results_dict:
            continue
            
        print(f" -> [{checkpoint_file}] Querying Gemini API for flow {i + 1} of {total}...")
        
        success = False
        while not success:
            try:
                start_time = time.perf_counter()
                
                # We use model.generate_content to get the JSON string, then parse it
                import json
                response = model.generate_content(text)
                result_data = json.loads(response.text)
                chosen_label = result_data.get("label")
                
                end_time = time.perf_counter()
                latency_sec = round(end_time - start_time, 3)
                
                if chosen_label not in threat_labels:
                    chosen_label = "Malformed protocol anomaly or unknown suspicious behavior"
                    
                results_dict[i] = (chosen_label, 1.0, latency_sec)
                success = True
                
                time.sleep(3) 
                
            except ResourceExhausted:
                print("[!] Rate limit hit (429). Sleeping for 60 seconds...")
                time.sleep(60)
                
            except Exception as e:
                print(f"[!] API Error on flow {i}: {e}")
                results_dict[i] = ("Malformed protocol anomaly or unknown suspicious behavior", 0.0, 0.0)
                success = True
        
        # --- 2. ATOMIC PICKLE SAVING (CRASH PROOF) ---
        if i > 0 and i % 50 == 0:
            tmp_file = checkpoint_file + ".tmp"
            
            # Step A: Write to the temporary file safely
            with open(tmp_file, 'wb') as f:
                pickle.dump(results_dict, f)
                
            # Step B: Instantly overwrite the real file. 
            # If the computer crashes during Step A, the real checkpoint is safe!
            os.replace(tmp_file, checkpoint_file) 

    # Final atomic save when the loop completes
    tmp_file = checkpoint_file + ".tmp"
    with open(tmp_file, 'wb') as f:
        pickle.dump(results_dict, f)
    os.replace(tmp_file, checkpoint_file)
        
    return results_dict
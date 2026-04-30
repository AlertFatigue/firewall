# dataLoader.py
import pandas as pd
import json
import gc
import config

def load_data(file_path):
    print(f"Streaming massive JSON file: {file_path}")
    
    # 1. Stream the file line-by-line to avoid RAM crashes
    chunk_list = []
    current_chunk = []
    chunk_size = 20000  # Processes 20,000 alerts at a time
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                # CRITICAL RAM SAVER: Only keep actual alerts, throw away the raw flows/dns
                if record.get("event_type") == "alert":
                    current_chunk.append(record)
            except json.JSONDecodeError:
                continue
            
            # When the chunk gets full, flatten it to a DataFrame and save it
            if len(current_chunk) >= chunk_size:
                df_chunk = pd.json_normalize(current_chunk)
                chunk_list.append(df_chunk)
                current_chunk = [] # Clear RAM
    
    # Catch any remaining alerts at the end of the file
    if len(current_chunk) > 0:
        df_chunk = pd.json_normalize(current_chunk)
        chunk_list.append(df_chunk)
        current_chunk = []
        
    print(f"Found {len(chunk_list)} chunks of alerts. Combining...")
    df = pd.concat(chunk_list, ignore_index=True)
    
    # Flush the chunk list from RAM entirely
    del chunk_list
    gc.collect()
    
    print(f"Total alerts extracted: {len(df)}")

    # 2. Extract Human Readable DF (For the LLM context later)
    # These are the columns classifier.py needs to build the text paragraph
    human_cols = ['src_ip', 'dest_ip', 'dest_port', 'proto', 'app_proto', 
                  'alert.signature', 'alert.severity', 'alert.category', 
                  'flow.pkts_toserver', 'flow.bytes_toserver', 'flow.pkts_toclient', 'flow.bytes_toclient']
    
    # Add dynamic HTTP/DNS/TLS columns if they happen to exist in the dataframe
    optional_cols = ['http.hostname', 'http.http_method', 'http.url', 'http.http_user_agent', 
                     'dns.rrname', 'dns.rcode', 'tls.sni', 'tls.subject']
    
    for col in optional_cols:
        if col in df.columns:
            human_cols.append(col)
            
    # Only select columns that actually exist so Pandas doesn't throw a KeyError
    valid_human_cols = [col for col in human_cols if col in df.columns]
    human_readable_df = df[valid_human_cols].copy()

    # 3. Create ML Ready DF
    # Drop columns explicitly marked in your config.py
    cols_to_drop = [col for col in config.COLS_TO_DROP if col in df.columns]
    ml_ready_df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return ml_ready_df, human_readable_df
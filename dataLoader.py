# dataLoader.py
import pandas as pd
import json

def load_data(file_path):
    parsed_json_list = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    # parse json string into py dict
                    event = json.loads(line)
                    
                    # making sure dns features properly extracted
                    if 'dns' in event and 'queries' in event['dns']:
                        queries = event['dns']['queries']
                      
                        if isinstance(queries, list) and len(queries) > 0:
                            if 'rrname' in queries[0]:
                                # flatten it
                                event['dns']['rrname'] = queries[0]['rrname']
                    
                    # append
                    parsed_json_list.append(event)
                    
                except json.JSONDecodeError:
                    # skip any corrupted
                    continue

    # normalize in to flat dataframe
    ml_ready_df = pd.json_normalize(parsed_json_list)
    
    # group by flow id
    ml_ready_df = ml_ready_df.groupby('flow_id').first()
    
    human_readable_df = ml_ready_df.copy()
    
    return ml_ready_df, human_readable_df
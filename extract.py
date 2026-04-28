import json
import os

def extract_data():
    # UPDATE THIS to your actual folder name if it's different!
    input_file = './trueHeartbleedLogs/eve.json' 
    output_file = './filtered_eve.json'
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}. Please check your folder name.")
        return

    print(f"Filtering {input_file} to save memory...")
    
    normal_count = 0
    max_normal_events = 15000 
    saved_lines = []

    with open(input_file, 'r') as infile:
        for line in infile:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # 1. ALWAYS grab the Heartbleed flows (using the true ports)
            is_heartbleed = (
                event.get('src_ip') == '172.16.0.1' and 
                event.get('dest_ip') == '192.168.10.51' and 
                event.get('dest_port') == 444
            )
            
            # 2. Grab normal flows until we hit 15,000
            is_normal = False
            if not is_heartbleed and normal_count < max_normal_events and event.get('event_type') == 'flow':
                is_normal = True
                normal_count += 1
                
            # 3. Save the exact original JSON line!
            if is_heartbleed or is_normal:
                saved_lines.append(line)

    # Write the exact same JSON format to a smaller file
    with open(output_file, 'w') as outfile:
        for line in saved_lines:
            outfile.write(line)

    print(f"Extraction complete! Saved {len(saved_lines)} events to {output_file}")

if __name__ == "__main__":
    extract_data()
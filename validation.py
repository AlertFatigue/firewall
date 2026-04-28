import pandas as pd
import joblib

def main():
    print("Re-linking IPs and Ports for validation...")
    
    # 1. Load the labeled data we just generated
    labeled_df = pd.read_csv('catboost_training_data.csv', index_col=0)
    
    # 2. Load the original human readable data (this has the IPs/Ports!)
    from dataLoader import load_data
    import config
    _, original_df = load_data(config.FILE_PATH)
    
    # 3. Join them back together using the index
    # This keeps the 15,426 rows but adds the 'src_ip', 'dest_ip', etc.
    validation_df = labeled_df.join(original_df[['src_ip', 'dest_ip', 'src_port', 'dest_port']])
    
    # 4. Save this as your "Master Validation" file
    validation_df.to_csv('master_validation_with_ips.csv')
    print("Done! 'master_validation_with_ips.csv' created.")

if __name__ == "__main__":
    main()

import os
import shutil
import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
import glob

# Paths
BASE_DIR = r'd:\Development\CODA\Bareq\VisionTera'
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
DATA_DIR = os.path.join(DATASETS_DIR, 'data')
MAT_FILE = os.path.join(DATASETS_DIR, 'annotation.mat')

NEW_DATA_ROOT = os.path.join(BASE_DIR, 'data')
NEW_MALE_DIR = os.path.join(NEW_DATA_ROOT, 'male')
NEW_FEMALE_DIR = os.path.join(NEW_DATA_ROOT, 'female')

CSV_FILES = {
    'train': os.path.join(DATASETS_DIR, 'train.csv'),
    'test': os.path.join(DATASETS_DIR, 'test.csv'),
    'val': os.path.join(DATASETS_DIR, 'val.csv')
}

def load_data():
    print("Loading existing data...")
    mat_data = scipy.io.loadmat(MAT_FILE, squeeze_me=False)
    dfs = {k: pd.read_csv(v) for k, v in CSV_FILES.items()}
    return mat_data, dfs

def get_new_images():
    print("Scanning for new images...")
    new_images = [] # List of (path, label_vector)
    
    # Label mapping: 'Female' is index 0. 
    # If male: index 0 = 0. If female: index 0 = 1.
    # All other 25 attributes set to 0 (unknown/default).
    
    for gender_dir, is_female in [(NEW_MALE_DIR, 0), (NEW_FEMALE_DIR, 1)]:
        if not os.path.exists(gender_dir):
            print(f"Warning: Directory not found: {gender_dir}")
            continue
            
        files = glob.glob(os.path.join(gender_dir, '*.*'))
        files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(files)} images in {gender_dir}")
        
        for fpath in files:
            fname = os.path.basename(fpath)
            label = np.zeros((26,), dtype=np.float64) # Float to match typical MAT/CSV types often
            # Actually check existing types later, but usually int or float. 
            # CSV shows 0 (int). MAT shows doubles usually in matlab.
            label[0] = is_female
            new_images.append({'path': fpath, 'filename': fname, 'label': label})
            
    return new_images

def resolve_duplicates(new_images, existing_names):
    print("Checking for duplicates...")
    final_images = []
    existing_set = set(existing_names)
    
    count_renamed = 0
    
    for item in new_images:
        original_name = item['filename']
        name, ext = os.path.splitext(original_name)
        
        new_name = original_name
        counter = 1
        
        while new_name in existing_set:
            new_name = f"{name}_new_{counter}{ext}"
            counter += 1
            
        if new_name != original_name:
            count_renamed += 1
            
        item['final_name'] = new_name
        existing_set.add(new_name)
        final_images.append(item)
        
    print(f"Renamed {count_renamed} files to avoid collisions.")
    return final_images

def update_mat_arrays(mat_data, key_prefix, new_items):
    # key_prefix is 'train', 'test', or 'val'
    # keys are: {prefix}_images_name, {prefix}_label
    
    name_key = f"{key_prefix}_images_name"
    label_key = f"{key_prefix}_label"
    
    # Get existing
    # images_name is typically (N, 1) array of objects (strings)
    # label is (N, 26)
    
    existing_names = mat_data[name_key]
    existing_labels = mat_data[label_key]
    
    print(f"  Existing {key_prefix}: {existing_names.shape}, {existing_labels.shape}")
    
    if not new_items:
        return mat_data
        
    # Prepare new arrays
    new_names_list = np.array([item['final_name'] for item in new_items], dtype=object).reshape(-1, 1)
    new_labels_list = np.array([item['label'] for item in new_items])
    
    # Concatenate
    updated_names = np.concatenate([existing_names, new_names_list], axis=0)
    updated_labels = np.concatenate([existing_labels, new_labels_list], axis=0)
    
    mat_data[name_key] = updated_names
    mat_data[label_key] = updated_labels
    
    print(f"  Updated {key_prefix}: {updated_names.shape}, {updated_labels.shape}")
    return mat_data

def update_csv(df, new_items, attributes):
    # df columns: Image, Female, AgeOver60, ...
    
    new_rows = []
    for item in new_items:
        row = {'Image': item['final_name']}
        # attributes[0] is Female. attributes corresponds to label indices.
        for idx, attr_name in enumerate(attributes):
            row[attr_name] = int(item['label'][idx])
            
        new_rows.append(row)
        
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Ensure column order matches
        new_df = new_df[df.columns]
        updated_df = pd.concat([df, new_df], ignore_index=True)
        return updated_df
    return df

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    mat_data, dfs = load_data()
    
    # Extract attributes list from mat data for CSV mapping
    # 'attributes' is (26, 1) cell array
    raw_attrs = mat_data['attributes']
    attributes = []
    for i in range(raw_attrs.shape[0]):
        try:
            val = raw_attrs[i][0][0]
            attributes.append(str(val))
        except:
             attributes.append(str(raw_attrs[i][0]))
             
    print(f"Attributes detected: {len(attributes)}")
    # sanity check
    if 'Female' not in attributes:
        print("CRITICAL Warning: 'Female' not found in attributes!")
    
    # Collect all existing names to check duplicates
    all_existing_names = set()
    for key in ['train_images_name', 'test_images_name', 'val_images_name']:
        names = mat_data[key]
        for i in range(names.shape[0]):
            try:
                all_existing_names.add(str(names[i][0]))
            except:
                pass
                
    new_images = get_new_images()
    if not new_images:
        print("No new images found. Exiting.")
        return

    new_images = resolve_duplicates(new_images, all_existing_names)
    
    # Split
    train_items, test_val_items = train_test_split(new_images, train_size=0.8, random_state=42)
    val_items, test_items = train_test_split(test_val_items, train_size=0.5, random_state=42)
    
    print(f"New split: Train={len(train_items)}, Val={len(val_items)}, Test={len(test_items)}")
    
    # Move files
    print("Moving files...")
    for item in new_images:
        src = item['path']
        dst = os.path.join(DATA_DIR, item['final_name'])
        shutil.copy2(src, dst) # copy2 preserves metadata, safer than move for now
        # os.remove(src) # Uncomment to actually move (delete source)
        
    # Update Data Structures
    print("Updating structures...")
    
    # Train
    mat_data = update_mat_arrays(mat_data, 'train', train_items)
    dfs['train'] = update_csv(dfs['train'], train_items, attributes)
    
    # Val
    mat_data = update_mat_arrays(mat_data, 'val', val_items)
    dfs['val'] = update_csv(dfs['val'], val_items, attributes)
    
    # Test
    mat_data = update_mat_arrays(mat_data, 'test', test_items)
    dfs['test'] = update_csv(dfs['test'], test_items, attributes)
    
    # Save
    print("Saving files...")
    scipy.io.savemat(MAT_FILE, mat_data, do_compression=True)
    
    for k, df in dfs.items():
        df.to_csv(CSV_FILES[k], index=False)
        
    print("Done!")

if __name__ == "__main__":
    main()

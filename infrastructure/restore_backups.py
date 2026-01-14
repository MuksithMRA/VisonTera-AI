
import os
import shutil

BASE_DIR = r'd:\Development\CODA\Bareq\VisionTera Project\VisionTera AI\datasets'
CSV_FILES = ['train.csv', 'val.csv', 'test.csv']

def restore_backups():
    for csv_file in CSV_FILES:
        backup_path = os.path.join(BASE_DIR, csv_file + ".bak")
        target_path = os.path.join(BASE_DIR, csv_file)
        
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, target_path)
            print(f"Restored {csv_file} from backup.")
            # os.remove(backup_path) # Keep backup just in case
        else:
            print(f"Backup not found for {csv_file}")

if __name__ == "__main__":
    restore_backups()

import sys
import shutil
import time

def copy_file_periodically(source, destination, t):
    while True:
        try:
            shutil.copy(source, destination)
            # print(f"File copied from {source} to {destination}")
        except Exception as e:
            # print(f"Error occurred while copying file: {e}")
            pass

        time.sleep(t)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py source_path destination_path [copy_interval]")
        sys.exit(1)

    source_path = sys.argv[1]
    destination_path = sys.argv[2]
    copy_interval = int(sys.argv[3]) if len(sys.argv) >= 4 else 60

    copy_file_periodically(source_path, destination_path, copy_interval)

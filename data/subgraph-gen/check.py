import os
import subprocess
import shutil

# TO CHECK IF THE RANDOM SEED WORKS

def run_command(command):
    result = subprocess.run(command, check=True, capture_output=False, shell=True)

folders = os.listdir() # Get a list of all folders in the current directory
for folder in folders:
    if folder in ["atomic", "graph-writer", "qagnn", "text2kg"]:
        continue
    if os.path.isdir(folder):
        print(f"Working on {folder}")
        command1 = f"cd {folder} && python convert.py"
        run_command(command1)
        
        old_proc_name = "proc"
        new_proc_name = "proc-temp"
        os.renames(os.path.join(folder, old_proc_name), os.path.join(folder, new_proc_name)) # Rename `proc` to `proc-temp`

        command2 = f"cd {folder} && python convert.py"
        run_command(command2)

        proc_files = os.listdir(os.path.join(folder, new_proc_name)) # List of files in `proc-temp`
        
        for file in proc_files:
            orig_file_content = open(os.path.join(folder, new_proc_name, file), 'r', encoding="utf-8").read()
            converted_file_content = open(os.path.join(folder, old_proc_name, file), 'r', encoding="utf-8").read()
            
            assert orig_file_content == converted_file_content, \
                f"Content mismatch between original and converted files ({new_proc_name}/{file})!"
                
        shutil.rmtree(os.path.join(folder, new_proc_name)) # Remove `proc-temp` directory
        shutil.rmtree(os.path.join(folder, old_proc_name)) # Remove `proc-temp` directory
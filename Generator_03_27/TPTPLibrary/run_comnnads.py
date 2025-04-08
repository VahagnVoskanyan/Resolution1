import subprocess
import os

output_dir = r'C:\\Users\\vahag\\Desktop\\BaseFolder\\University\\Thesis\\GitFiles1\\Generator_03_27\\TPTPLibrary\\Output1'
os.makedirs(output_dir, exist_ok=True)

command = [
    "docker", "run", "--rm", "-it",
    "-v", f"{output_dir}:/vampire/examples/Output",
    "--name", "vampire_clausify", "vampire",
    "/bin/bash", "-c",
    (
        'for f in /vampire/examples/Problems1/*.p; do '
        'base=$(basename "$f" .p); '
        './vampire --proof_extra full -t 100 "$f" > /vampire/examples/Output/"${base}"_result.txt; '
        'done'
    )
]

# Run the command
subprocess.run(command)

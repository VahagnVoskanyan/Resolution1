import subprocess
import os

output_dir = r'C:\\Users\\vahag\\Desktop\\BaseFolder\\University\\Thesis\\GitFiles1\\Generator_04_09\\Axioms_clausified'
os.makedirs(output_dir, exist_ok=True)

command = [
    "docker", "run", "--rm", "-it",
    "-v", f"{output_dir}:/vampire/examples/Output",
    "--name", "vampire_clausify", "vampire",
    "/bin/bash", "-c",
    (
        'for f in /vampire/examples/Axioms/*.ax; do '
        'base=$(basename "$f" .p); '
        './vampire --mode clausify -t 100 "$f" > /vampire/examples/Output/"${base}"_claused.txt; '
        'done'
    )
]

# Run the command
subprocess.run(command)

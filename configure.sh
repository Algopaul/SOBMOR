#!/bin/bash

# Output file
output_file="runner_definition.mk"

echo "Checking for SLURM installation"

if command -v srun &> /dev/null; then
    # Prompt the user for any extra flags
    echo "SLURM installation found. Running commands with srun --time 3:00:00 --mem 32G --cpus-per-task=32 --ntasks=1"
    read -p "Please enter any extra flags for srun: " extra_flags

    # If srun is available, set the RUN variable accordingly
    echo "RUN=srun $extra_flags --time 3:00:00 --mem 32G --cpus-per-task=32 --ntasks=1" > "$output_file"

    # Prompt the user for their email
    read -p "Please enter your email to get notified when your jobs complete: " email


    # Write the extra target into output_file
    echo "run_all:" >> "$output_file"
    echo -e "\t-bash run.sh" >> "$output_file"
    echo -e "\tsrun  --job-name \"SOBMOR jobs complete\" --mail-type END --mail-user $email $extra_flags ls -a"  >> "$output_file"
else
    echo "No SLURM installation found, running commands without srun"
    # If srun is not available, set the RUN variable to an empty string
    echo 'RUN=' > "$output_file"
fi

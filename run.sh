#python3 -c "import torch; print(torch.__version__)"
#echo "Currently active Conda environment: $CONDA_DEFAULT_ENV"

command="python3 tryon_dress/test_start.py"
#command="python3 tryon_dress/test_start.py"
while true; do
    # Run the command
    $command

    # Check the exit status
    if [ $? -eq 0 ]; then
        # If the command succeeds, break out of the loop
        break
    else
        # If the command fails, print an error message and wait for some time before retrying
        echo "Command failed with exit code $?. Retrying in 5 seconds..."
        sleep 5  # Adjust the sleep duration as needed
    fi
done

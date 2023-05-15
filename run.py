import subprocess

# Run the first program
subprocess.run(["python", "harm.py"])

# Run the second program after the first program has finished executing
subprocess.run(["python", "video.py"])

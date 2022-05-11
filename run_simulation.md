# Workflow of the simulation

1. run `./run_auto.py`
This is generating run_0, ..., run_N folders inside `./csv_files/`
2. Move the run_n files into a new folder inside `./csv_files`
3. Change the folder directory inside the python file: `./analyze_auto.py`
4. run `./analyze_auto.py`
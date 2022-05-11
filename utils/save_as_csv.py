import pandas as pd
import os

my_path = os.path.join('D:', os.sep, 'local_github', 'particles_nir_repo', 'csv_files')

pd_target = pd.DataFrame(target)
pd_output = pd.DataFrame(output)

pd_target.to_csv(os.path.join(my_path, "target.csv"))
pd_output.to_csv(os.path.join(my_path, "output.csv"))


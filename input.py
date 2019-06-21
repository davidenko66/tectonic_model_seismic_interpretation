# Example parameter file to be used with data_converstion.py

# Data directory
vtk_directory = '/home/naliboff/projects/tsmi/models/tmsi_t1/output/solution/'

# Results directory
results_directory = '/home/naliboff/projects/tsmi/models/tmsi_t1/output/processing/'

# Type of data to import (options: 'vtk', 'npz')
data_import = 'vtk'

# Flag indicating whether (yes) or not (no) to make plots
plot_results = 'yes'

# Generate array with time steps
first_time_step  = 298
last_time_step   = 298

time_step_interval = 1

# Model resolution
gres = 0.4e3

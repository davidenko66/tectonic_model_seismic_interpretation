# Code is executed with "python data_conversion.py" or "ipython data_conversion.py" 
# Load modules
import numpy as np

# Get user input parameters, which must be in a file called input.py located in this folder
exec('from input import *') 

# Array containing all time step numbers to analyze
time_steps  = np.arange(first_time_step,last_time_step+1,time_step_interval)

def main():

  # Loop through time steps
  for t in range(time_steps.size):

    # Extract numpy arrays from vtk data or load existing data
    if data_import == 'vtk':
      x, y, z, density, plastic_strain = get_numpy_arrays(t)
    else:
      x, y, z, density, plastic_strain = load_numpy_arrays(t)

    # Make plots
    if plot_results == 'yes':
      plot_data(x, y, density, plastic_strain, t, gres)

#------------------------------------------------------------------------------

def get_pvtu_number(time_step):
  
  if time_step<10:
    pvtu_number = '0000' + str(time_step)
  elif time_step>=10 and time_step<100:
    pvtu_number = '000' + str(time_step)
  elif time_step>=100 & time_step<1000:
    pvtu_number = '00' + str(time_step)

  return pvtu_number

#------------------------------------------------------------------------------

def load_numpy_arrays(t):

  pvtu_number = get_pvtu_number(time_steps[t])

  unfiltered = np.load(results_directory + '/' + 'unfiltered_' + pvtu_number + '.npz')

  xL = unfiltered['x']; yL = unfiltered['y']; zL = unfiltered['z']; dL = unfiltered['density']; 

  return xL, yL, zL, dL 

#------------------------------------------------------------------------------

def get_numpy_arrays(t):

  # Load modules
  import vtk as vtk; from vtk.util import numpy_support

  # Get pvtu number
  pvtu_number = get_pvtu_number(time_steps[t])

  # Load vtu data (pvtu directs to vtu files)
  reader = vtk.vtkXMLPUnstructuredGridReader()
  reader.SetFileName(vtk_directory + 'solution-' + pvtu_number + '.pvtu')
  reader.Update()

  # Get the coordinates of nodes in the mesh
  nodes_vtk_array= reader.GetOutput().GetPoints().GetData()

  # Convert nodal vtk data to a numpy array
  nodes_numpy_array = vtk.util.numpy_support.vtk_to_numpy(nodes_vtk_array)

  # Extract x, y and z coordinates from numpy array 
  x,y,z= nodes_numpy_array[:,0] , nodes_numpy_array[:,1] , nodes_numpy_array[:,2]

  # Determine the number of scalar fields contained in the .pvtu file
  number_of_fields = reader.GetOutput().GetPointData().GetNumberOfArrays()

  # Determine the name of each field and place it in an array.
  field_names = []
  for i in range(number_of_fields):
    field_names.append(reader.GetOutput().GetPointData().GetArrayName(i))

  # Determine the index of the field strain_rate
  idx = field_names.index("strain_rate")

  # Extract values of strain_rate
  field_vtk_array = reader.GetOutput().GetPointData().GetArray(idx)
  strain_rate     = numpy_support.vtk_to_numpy(field_vtk_array)

  # Determine the index of the field plastic_yielding
  idx = field_names.index("plastic_yielding")

  # Extract values of plastic yielding
  field_vtk_array  = reader.GetOutput().GetPointData().GetArray(idx)
  plastic_yielding = numpy_support.vtk_to_numpy(field_vtk_array)

  # Determine the index of the field density
  idx = field_names.index("density")

  # Extract values of plastic yielding
  field_vtk_array  = reader.GetOutput().GetPointData().GetArray(idx)
  density          = numpy_support.vtk_to_numpy(field_vtk_array)

  # Determine the index of the field plastic_strain
  idx = field_names.index("plastic_strain")

  # Extract values of plastic yielding
  field_vtk_array  = reader.GetOutput().GetPointData().GetArray(idx)
  plastic_strain   = numpy_support.vtk_to_numpy(field_vtk_array)


  # Save unfiltered (raw vtk) arrays
  np.savez(results_directory + '/' + 'unfiltered_' + pvtu_number, x=x, y=y, z=z, density=density)

  return x, y, z, density, plastic_strain

#------------------------------------------------------------------------------

def plot_data(x, y, density, plastic_strain, t, gres):

  # Load modules
  import matplotlib.pyplot as plt
  import matplotlib.colors as colors
  from matplotlib import ticker
  from scipy.interpolate import griddata

  # Define min and max x-y values
  xmin, xmax = min(x), max(x)
  ymin, ymax = min(y), max(y)

  print(xmin, xmax,ymin,ymax)

  # Number of points in x and y directions
  Nx = x.shape[0]
  Ny = y.shape[0]

  # Define grid spacing
  xi = np.linspace( xmin, xmax, int((xmax - xmin)/gres) + 1)
  yi = np.linspace( ymin, ymax, int((ymax - ymin)/gres) + 1)

  print(xi.min(),xi.max(),yi.min(),yi.max())
  print(x.min(),x.max(),y.min(),y.max())

  X, Y = np.meshgrid(xi, yi)

  density_grid = griddata((x, y), density, (xi[None,:], yi[:,None]), method='cubic')
  plastic_strain_grid = griddata((x, y), plastic_strain, (xi[None,:], yi[:,None]), method='cubic')
  
  print(np.min(density_grid),np.max(density_grid))
  print(density_grid.min(),density_grid.max())
  print(plastic_strain_grid.min(),plastic_strain_grid.max())

  print(density.min(),density.max())
  print(plastic_strain.min(),plastic_strain.max())

  # Initialize figure
  fig, ax = plt.subplots(2,1)

  # Plot density
  pcm = ax[0].pcolormesh(X[:,:]/1.e3,Y[:,:]/1.e3,density_grid,
                         norm=colors.Normalize(vmin=density_grid.min(), vmax=density_grid.max()),
                         cmap='coolwarm')
  fig.colorbar(pcm, ax=ax[0], extend='max',orientation='horizontal',label='Density (kg/m^3)',
               pad=0.30, shrink=0.5)
  ax[0].set_xlabel('Horizontal Position (km)')
  ax[0].set_ylabel('Vertical Position (km)') 
  ax[0].set_aspect(1)

  # Plot plastic strain
  pcm = ax[1].pcolormesh(X[:,:]/1.e3,Y[:,:]/1.e3,plastic_strain_grid,
                         norm=colors.Normalize(vmin=plastic_strain_grid.min(), vmax=plastic_strain_grid.max()),
                         cmap='coolwarm')
  fig.colorbar(pcm, ax=ax[1], extend='max',orientation='horizontal',label='Plastic Finite Strain Invariant)',
               pad=0.30, shrink=0.5)
  ax[1].set_xlabel('Horizontal Position (km)')
  ax[1].set_ylabel('Vertical Position (km)')    
  ax[1].set_aspect(1)

  # Get pvtu number
  pvtu_number = get_pvtu_number(time_steps[t])

  # Define name for figure
  figure_name = results_directory + '/' + 'density_plastic_strain' + '_' + pvtu_number + '.png'

  # Cleanup and save figure 
  fig.tight_layout() 
  plt.savefig(figure_name,dpi=300)

#------------------------------------------------------------------------------

# Call main function
main()

"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: nc2h5.py
@time: 2021/5/22 10:20
@desc:
"""
import netCDF4 as nc
import h5py
import os
ncfile=r'C:\git_repository\common/morel_fq.nc'
h5file=os.path.dirname(ncfile)+os.sep+os.path.basename(ncfile)[0:-3]+'.h5'
obj=nc.Dataset(ncfile)
h5=h5py.File(h5file,'w')
for attrname in obj.ncattrs():
    h5.attrs[attrname] = getattr(obj, attrname)

h5.attrs['add_description'] = 'this h5 file is convert from the .nc file by lwk1542@hotmail.com'

for name,variable in obj.variables.items():
     vari=h5.create_dataset(name, data=variable[()].data)
     for attrname in variable.ncattrs():
         vari.attrs[attrname] = getattr(variable, attrname)

h5.close()
obj.close()
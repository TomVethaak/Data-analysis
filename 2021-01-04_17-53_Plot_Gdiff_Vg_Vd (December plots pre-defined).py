from __future__ import print_function

# Select the user that is running the code
User            = "Tom"
#User           = "Francois"

# D63D3
DataDirectoryD63D3S04   = "2020-11 PtSi D63/D3/Base Temp/D3_Gdiff_Vd=+-1(0.01)mV_Vac=10uV_Vg=-1V_-4.5V(0.01)/"
DataDirectoryD63D3S11   = "2020-11 PtSi D63/D3/Base Temp/H=100mT/D3_Gdiff_Vd=+-1(0.01)mV_Vac=10uV_Vg=-4.5_-1(0.01)V/"
DataDirectoryD63D3S12   = "2020-11 PtSi D63/D3/Tsweep Vg=-4.5V H=100mT/"
DataDirectoryD63D3S13   = "2020-11 PtSi D63/D3/Tsweep Vg=-2.7V H=100mT/"
DataDirectoryD63D3S14   = "2020-11 PtSi D63/D3/Base Temp/D3_Gdiff_Vd=+-1(0.01)mV_Vac=10uV_Vg=-4.9V_-4.0V(0.01)/"

# D63D4
DataDirectoryD63D4S01   = "2020-11 PtSi D63/D4/Base temp/D4_Gdiff_Vd=+-1(0.01)mV_Vac=10uV_Vg=-4.0V_-1.0V(0.01)/"
DataDirectoryD63D4S02   = "2020-11 PtSi D63/D4/Base temp/D4_Gdiff_Vd=+-1(0.01)mV_Vac=10uV_Vg=-2.2V_-1.0V(0.0002)/"
DataDirectoryD63D4S03   = "2020-11 PtSi D63/D4/Base temp/D4_Gdiff_Vd=+-5(0.01)mV_Vac=10uV_Vg=-4.0V_-1.0V(0.01)/"

DataDirectory   = DataDirectoryD63D4S03
SweepType       = "VgVd"    # VgVd, HVd, TVd
# The X axis can be automatically extracted from VgVd file names
# For H or T sweeps all we get in the file names is #0001, #0002 etc
SweepStart      = 1         # The H or T value of the first file (#0001)
SweepStep       = -0.05     # The H or T step in Labview units

# At higher lock-in frequencies we get a shift in phase.
PhaseShift      = 5.2       # degrees
RFilters        = 41500     # Ohm
VdOffset        = -0.035    # mV

#%% INITIALIZE
# This cell is for initialization: import the relevant libraries/modules, choose what directory we are importing things from

import guidata                        # guidata
from guiqwt.plot import ImageDialog   # ImageDialog does ???
from guiqwt.builder import make       # make does ???
import numpy as np                    # numpy gives us numerical analysis functions
import os                             # os, "miscellaneous operating system interfaces", is for handling directory/file operations. Useful for getting the files in a directory
import re                             # re for regular expressions (https://docs.python.org/3/howto/regex.html)
from scipy.interpolate import interp1d

_app        = guidata.qapplication()

def remove_filters( _DataArray ):
    # Create an empty array of the same size as the original data array,
    # - for the adjusted values:
    AdjustedDataMap         = np.zeros( ( len( _DataArray ) , len ( _DataArray[0] ) ,
                                      len( _DataArray[0,0] ) ) )
    # - and for the interpolated values, with ten times smaller Vd steps: 
    InterpolatedData        = np.zeros( ( len( _DataArray ) , 
                                          len( _DataArray[0] ) * 10,
                                          len( _DataArray[0,0] ) ) )
    for VgIterator in range( 0 , len( _DataArray ) ):
        for VdIterator in range( 0 , len( _DataArray[0] ) ):
            # Values given in the data files
            _Vd             = _DataArray[ VgIterator , VdIterator , 0 ] - VdOffset
            _Id             = _DataArray[ VgIterator , VdIterator , 1 ]
            _Gdiff          = _DataArray[ VgIterator , VdIterator , 2 ] * np.cos( 
                              ( _DataArray[ VgIterator , VdIterator , 3 ] - 
                              PhaseShift ) * np.pi / 180 )
            _GdiffPhase     = _DataArray[ VgIterator , VdIterator , 3 ]
            _ILeak          = _DataArray[ VgIterator , VdIterator , 4 ]
            _Vg             = _DataArray[ VgIterator , VdIterator , 5 ]
            _T              = _DataArray[ VgIterator , VdIterator , 6 ]
            
            # Adjusted values
            _GdiffDUT       = 1 / ( 1 / _Gdiff - RFilters )
            _VdDUT          = _Vd - 0.001 * RFilters * _Id  # Factor 0.001: Vd is in mV
            
            # Return the 
            AdjustedDataMap[ VgIterator , VdIterator , 0:7 ] = [ _VdDUT , _Id ,
                                _GdiffDUT , _GdiffPhase , _ILeak , _Vg , _T ]
        
    # The Vd values in AdjustedDataMap are no longer equally spaced.
    # We are going to interpolate between them, and then evaluate that
    # interpolation at evenly spaced Vd values.

    # The Vg values (column 5) of the output are the same as the input
    InterpolatedData[:,:,5] = np.array( [ _DataArray[:,0,5] ] * 
                                len( InterpolatedData[0] ) ).T
    # The output Vd range is given by AdjustedDataMap, and has more points
    InterpolationRangeMin   = min( [ min( AdjustedDataMap[VgIterator,:,0] ) for VgIterator
                                     in range( 0 , len(AdjustedDataMap) ) ] )
    InterpolationRangeMax   = max( [ max( AdjustedDataMap[VgIterator,:,0] ) for VgIterator
                                     in range( 0 , len(AdjustedDataMap) ) ] )
    InterpolatedData[:,:,0] = np.linspace(  InterpolationRangeMin ,
                                            InterpolationRangeMax ,
                                            num=len( InterpolatedData[0] ) ,
                                            endpoint=True )
    # The other columns will be interpolated
    for VgIterator in range( 0 , len( _DataArray ) ):
        AdjustedVd  = AdjustedDataMap[ VgIterator , : , 0 ]
        for Column in [1, 2, 3, 4, 6]:
            AdjustedColumn                  = AdjustedDataMap[ VgIterator , : , Column ]
            InterpolatedFunction            = interp1d( AdjustedVd , AdjustedColumn ,
                                                       kind='nearest',
                                                       bounds_error = False ,
                                                       fill_value=0)
            InterpolatedData[ VgIterator , 
                             : ,Column]     = InterpolatedFunction( 
                                                         InterpolatedData[0,:,0] )
    return InterpolatedData

def groupby_mean(a):
    # Sort array by groupby column
    b = a[a[:,0].argsort()]
    # Get interval indices for the sorted groupby col
    idx = np.flatnonzero(np.r_[True,b[:-1,0]!=b[1:,0],True])
    # Get counts of each group and sum rows based on the groupings & hence averages
    counts = np.diff(idx)
    avg = np.add.reduceat(b[:,1:],idx[:-1],axis=0)/counts.astype(float)[:,None]
    # Finally concatenate for the output in desired format
    return np.c_[b[idx[:-1],0],avg]

def imshow( x, y, data ):
    if SweepType == "VgVd":
        XLabel  = "Gate Voltage (V)"
    elif SweepType == "TVd":
        XLabel  = "T (K)"
    elif SweepType == "HVd":
        XLabel  = "H (mT)"
    else:
        print("Error: SweepType not found")
        exit
    win = ImageDialog(edit=False, toolbar=True,
                      wintitle="Image with custom X/Y axes scales",
                      options=dict(xlabel=XLabel, ylabel="Drain Voltage (mV)",
                                   yreverse=False,show_contrast=True))
    item = make.xyimage(x, y, data)
    plot = win.get_plot()
    plot.set_aspect_ratio(lock=False) 
    plot.add_item(item)
    win.show()
    win.exec_()
     
if User == "Francois":
    # Francois stores the data on his C drive
    RootDirectory   = "C:/Users/FL134692/Documents/ANR - Projets/2019 SUNISIDEUP/Data/C2N_PtSi/"
elif User == "Tom":
    # Tom stores the data on his Z drive
    RootDirectory   = "Z:/Data/Data sorted by lot/W12 Laurie Calvet's wafer/Christophe's fridge/"
else:
    print("Error: User not found")
    exit

#%% IMPORT
# This cell is for importing the data, it takes the longest.

# Start by listing the files in the directory
DataFileList    = np.array( [ entry for entry in os.listdir( RootDirectory + DataDirectory )[1*bool(SweepType=="TVd" or SweepType=="HVd")::] ] )
DataArray       = np.array( [ np.loadtxt( RootDirectory + DataDirectory + Filename ) for Filename in DataFileList ])

# We are dealing with two naming systems: either Vg=-X.12345V_Y.dat, or Vg=-X.1234V_.dat.
# The most robust way of extracting the Vg value seems to find any expression between "Vg=" and "V"
# Let's use regular expressions: 
# 'DataFileList' is an array with strings.
# For each element 'filename' we use the regular expression "(?<=Vg=).*(?=V)" to find all numbers that are sandwiched between "Vg=" and "V"
# We then take the first (and only) element of that list, index [0], and convert it from a string to a float (a number that uses two bytes)
if SweepType == "VgVd":
    XValues     = np.array( [ float( re.findall( "(?<=Vg=).*(?=V)" , Filename )[0] ) for Filename in DataFileList ] )
elif SweepType == "TVd" or "HVd":
    XValues     = np.array( [ SweepStart + XIterator * SweepStep for XIterator in range( 0 , len( DataArray ) ) ] )
else:
    print("Error: SweepType not found")
    exit

# If the measurement started at larger Vg values, and then moved to smaller (or more negative) values, invert the VgValues and DataArray arrays
# imshow requires the arrays to be sorted from left to right.
if XValues[0]>XValues[-1]:
    XValues     = XValues[::-1]
    DataArray   = DataArray[::-1]

#%% PLOT
# This cell is for manipulating and plotting the data
DataArrayAveraged       = np.array( [ groupby_mean( DataArray[ XIterator ] ) for XIterator in range( 0 , len( DataArray ) ) ] )
DataArrayAdjusted       = remove_filters( DataArrayAveraged )
# Gdiff without averaging over recurring Vd values:
Vd                      = DataArray[ 0 , :len( DataArray[0,:] ) / 2 , 0 ]
Gdiff                   = np.swapaxes( DataArray[ : , :len (DataArray[0,:] ) / 2 , 2 ] , 0 , 1 )
# Gdiff averaged over recurring Vd values:
VdAveraged              = DataArrayAveraged[ 0 , : , 0 ]
GdiffAveraged           = np.swapaxes( DataArrayAveraged[ : , : , 2 ] , 0 , 1 )
# Gdiff corrected for the filters:
VdAdjusted              = DataArrayAdjusted[ 0 , : , 0 ]
GdiffAdjusted           = np.swapaxes( DataArrayAdjusted[ : , : , 2 ] , 0 , 1 )

#imshow(VgValues,Vd,Gdiff)
#imshow(VgValues,VdAveraged,GdiffAveraged)
imshow(XValues,np.sort(VdAdjusted),GdiffAdjusted)
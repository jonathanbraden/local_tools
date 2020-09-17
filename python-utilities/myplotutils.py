#!/usr/bin/env python

####################################################################
# A collection of useful tools for making publication quality papers
####################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as coll
import matplotlib.cm as mplcm
import matplotlib.colors as colors

def myShow(fName='temp',Preview=True,makePDF=True,makePNG=True):
    """Clear the current matplotlib figure.  If Preview is True, will also display the figure before clearing it.  This is a helper function for automation of paper plot generation.  When generating many plots, use myShow() after every figure save.  When fine-tuning plots, set default of Preview to True to make sure they look the way you want.  When reproducing plots to adjust font sizes, set default of Preview to False and the figures will just be saved.

    Input:
      Preview (Boolean) - If True, show a preview of the figure before clearing.
                          If False, simply clears the figure without showing a preview.

    To Do: 
      Add stripping of erroneous file endings such as .pdf or .png in fName
    """
    dpiRes = 300

    if (makePDF):
        plt.savefig(fName+'.pdf')
    if (makePNG):
        plt.savefig(fName+'.png',dpi=dpiRes)
    if (Preview):
        plt.show()
    plt.clf()
    return

# To do: Add some choices for how rounding is done in the nDec<0 case.  Add some error checking
def sciNotLatex(x,nDec):
    """Return the value of x typeset in useful scientific notation with nDec digits of accuracy in the mantissa.  
    If nDec < 0, simply returns 10^power, with power rounded to the nearest integer

    Input:
      x (float) - value to return scientific notation of
      nDec (int) - Number of decimal places to retain

    Returns (String):
      A LaTeX typeset string of x.  The math environment '$' signs are not included.

    Sample use: 
      myLabel = r'$'+sciNotLatex(3.7151,1)+'$'
    """
    power = int( np.floor(np.log10(x)) )
    mant = x /10.**power
    if (mant > 10.):
        power = power + 1
        mant = mant / 10.

    if (nDec >= 0):
        tmp = r'{0:.'+str(nDec)+'f}'
        newString = tmp.format(mant)+r'\times'+r'10^{'+r'{0:}'.format(power)+r'}'
    else:
        if (np.log10(mant) > 0.5):
            power = power+1
        newString = r'10^{'+r'{0:}'.format(power)+r'}'
    return newString

def myColorbar(**kwargs):
    cb = plt.colorbar(**kwargs)
    cb.solids.set_rasterized(True)
    return cb

def removeSpines(ax):
    """ Remove the top and right spines of the given axis.
        For aesthetic reasons, the ticks associated with these axes are also removed

    Input : ax - A matplotlib axis instance
    """
    ax.spines['right'].set_visible(False); ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False); ax.xaxis.set_ticks_position('bottom')
    return

def fillContourLines(cnt):
    """ Set the borders of countour plot divisions to be the same color as the interior.  This removes the ugly aliasing issues associated with making PDF contour plots.
    """
    for c in cnt.collections:
        c.set_edgecolor("face")
    return

# Need to write this.  Add zorder and rasterisation
def contourf_raster(ax=None,*args,**kwargs):
    if ax == None:
        plt.contourf()

# c.f. http://stackoverflow.com/questions/36074455/python-matplotlib-with-a-line-color-gradient-and-colorbar
def multicolor_line(x,y,z=None,cmap='copper',norm=plt.Normalize(0.0,1.0),linewidth=1,alpha=1.0):
    """ Plot a line with color gradient either linearly increasing along the length of the line, or else colored according to a scalar value.
    
    Input:
      **kwargs : Any of the arguments that can be passed to LineCollection.  Common options are:
    """
    if z is None:
        z = np.linspace(0.,1.,len(x))
    if not hasattr(z,"__iter__"):  # Some strange hack I don't understand
        z = np.array([z])

    z = np.asarray(z)  # What does this do?
    segments = make_segments(x,y)
    lc = coll.LineCollection(segments,array=z,cmap=cmap,norm=norm,linewidth=linewidth,alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    # Now scale the axes.  Find a nicer way to do this.  Probably best to add a scaling flag.  Or else test current limits.
    pad=0.05
    plt.xlim((1.+pad)*x.min()-pad*x.max(),(1.+pad)*x.max()-pad*x.min())
    plt.ylim((1.+pad)*y.min()-pad*y.max(),(1.+pad)*y.max()-pad*y.min())
    
    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates to make a LineCollection object.
    Returns an array of the form numlines x (points per line) x 2 (x and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def makeColorCycle(myCmap,nLines):
    """Create a set of nLines colors generated from the colormap myCmap.

    Input:
      myCmap - a colormap name for use in matplotlib.pyplot.get_cmap()
      nLines (int) - The number of different line colors to return

    Returns (list of tuples):
      A list of colors in RGBA format
    """
    cm = plt.get_cmap(myCmap)
    colors = cm( np.linspace(0,1,nLines) )
    return colors

def newAxisLineColors(myCmap,nLines):
    """Return a figure with corresponding axis intended to hold nLines that cycle through the color map myCmap
    
    Input:
      myCmap () - Name of colormap to use
      nLines (int) - number of lines to cycle through

    Returns:
      [0] (matplotlib Figure) - The figure containing the axis
      [1] (matplotlib Axis) - The axis with correct line color cycle
    """
    fig = plt.figure()
    curax = fig.add_subplot(111)

    cm = plt.get_cmap(myCmap)
    cNorm = colors.Normalize(vmin=0, vmax=nLines-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    curax.set_color_cycle([scalarMap.to_rgba(i) for i in range(nLines)])
    return fig, curax

# Basically copy the function above here, except take an existing axis as input
def setAxisLineColors(curax,myCmap,nLines):
    return
#
# Give the plotted axes the same aspect ratio as the figure
# Curfig is the current figure, curax is the current axis
#
def fix_axes_aspect(curfig, curax, fix_to_fig=False, rat=0.5*(np.sqrt(5.)-1.) ):
    """ Set the aspect ratio for the axes of a figure to the given value.
    Default is the golden ratio R = \frac{\sqrt{5}-1}{2}.

    Input:
      curfig (matplotlib Figure) - 
        Handle to the figure containing the axis
      curax (matplotlib Axis) - 
        Handle to the axis object we want to resize
      fix_to_fig (Boolean) (optional, default=False) - 
        If True: the ratio of axis is set to the ratio of the figure (ie. final PDF including whitespace, axis labels, etc)
        If False: the ratio of the axis is set to the value of rat.
      rat (float) (optional, default = 0.5*(sqrt(5)-1) - 
        The ratio of the axis height to width.  Only used if (fix_to_fig == False)

    Returns:
      Null

    Important, this documentation may be incorrect.  Check that it actually does what it says it does

    TO DO: Check that this works with logarithmic axes
    """
    ax_rat=curax.get_data_ratio()
    if fix_to_fig:
        fig_size = curfig.get_size_inches()
        fig_rat = fig_size[1]/fig_size[0]
    else:
        fig_rat=rat
#    plt.axes().set_aspect(fig_rat/ax_rat,'box')
    curax.set_aspect(fig_rat/ax_rat,adjustable='box')  # What is this doing?

def get_ax_size(curax):
    """

    Input:
      curax (matplotlib Axis) - 
        The axis to get the dimensions of

    Returns:
      
    """
    bbox = curax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi

# This clearly isn't finished.  Presumably this is the one where I allow for things like fixing a given margin (for alignment in LaTeX)
def fix_axes_aspect_new(curfig,curax):
    plt.gca().set_position([0.15,0.15,0.95-0.15,0.95-0.15])


from matplotlib.collections import Collection
from matplotlib.artist import allow_rasterization
# Helper class to allow rasterisation of a contour plot
class ListCollection(Collection):
    def __init__(self, collections, **kwargs):
        Collection.__init__(self, **kwargs)
        self.set_collections(collections)
    def set_collections(self, collections):
        self._collections = collections
    def get_collections(self):
        return self._collections
    @allow_rasterization
    def draw(self, renderer):
        for _c in self._collections:
            _c.draw(renderer)

def insert_rasterized_contour_plot(c):
    """Rasterize the contours in a previously existing contour plot.  
    Leaves text such as axis labels in a vectorized format.
    
    Input (matplotlib ContourPlot Figure) - 
      c - A filled contour plot that should be rasterized
    Output (matplotlib ContourPlor Figure) -
      cc - The rasterized contour plot
    """
    collections = c.collections
    for _c in collections:
        _c.remove()
    cc = ListCollection(collections, rasterized=True)
    ax = plt.gca()
    ax.add_artist(cc)
    return cc

# To do, make the error checking better
def writeLinePlotData(xvals,yvals,fname):
    """Write out the data for producing a single matplotlib line as two columns.
    The first column contains the x data and the second the y data.
    
    Input :
      xvals - Vector with xvalues
      yvals - Vector with yvalues
      fname - Name of file to output data in
    """
    if (len(xvals) != len(yvals)):
        print "Error, number of x and y values must be equal"

    outString=''
    for i in range(len(xvals)):
        outString = outString + '{0:} {1:}\n'.format(xvals[i],yvals[i])
    fcur = open(fname,'w')
    fcur.write(outString)
    fcur.close()
    return

def makeNormalisedPDF(vals,w,nBins):
    mean = np.sum(vals) / len(vals)
    rms = np.sqrt( np.sum( (vals-mean)**2 ) / len(vals) )

    nVals = vals / rms
    pdf, binVals = np.histogram(nVals,bins=nBins,weights=w,density=True)

    return pdf, binVals

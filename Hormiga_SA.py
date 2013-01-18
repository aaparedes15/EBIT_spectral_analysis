#!/usr/bin/env python

# modules
import matplotlib.pyplot as plt
import numpy as np
import Tkinter
import tkMessageBox
import sys
import tkFileDialog
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import leastsq 

# globals
xdata = 0
ydata = 0
dx_input = 0
min_line_height = 0

def save_gauss_fits():
    global xdata
    global ydata
    pdf = PdfPages('fit_plots.pdf')
    big_fig = plt.figure(figsize=(9,5))
    big_ax = big_fig.add_subplot(111)
    big_ax.plot(xdata, ydata, c='b')
    big_ax.set_title("Full Spectrum")
    big_ax.set_xlabel("X-position")
    big_ax.set_ylabel("Counts")

    # empty list for data output
    y0 = []
    A  = []
    x0 = []
    sigma = []
    fwhm = []

    # use of find_lines to get x_edges
    global dx_input
    global min_line_height
    min_x = []
    max_x = []
    dx = float(dx_input.get())
    min_line_height = float(min_line_height_input.get())
    x_edges = find_lines(xdata, ydata, dx, min_line_height)['ranges']
    for i in x_edges:
        min_x.append(i[0])
        max_x.append(i[1])

    # by a loop, go through the spectral lines, fit them, and save the figures to the pdf
    figs = []
    for i in xrange(len(min_x)):
        # set the range
        good = (xdata > min_x[i])&(xdata < max_x[i])

        params, fig = gauss_fit(xdata[np.where(good)],
                                ydata[np.where(good)],
                                make_plot=True)

        # put the data in the list
        y0.append(params[0])
        A.append(params[1])
        x0.append(params[2])
        sigma.append(params[3])
        fwhm.append(2*np.sqrt(2*np.log(2))*params[3])

        # put some plot on big_fig
        x_data2 = np.linspace(min(xdata[np.where(good)]),max(xdata[np.where(good)]),num=150)
        def best_fit(x):
            return params[0] + params[1]*np.exp(-(x-params[2])**2 / params[3]**2)
        big_ax.plot(x_data2, best_fit(x_data2), c='r', alpha=0.5)	

        # save the figure
        figs.append(fig)

    big_ax.set_ylim(0,2000)
    big_ax.vlines(x0, 0, 2000, colors='k', linestyles='dotted', alpha=0.5, lw=0.3)
    for fig in figs:
        pdf.savefig(fig)
    pdf.savefig(big_fig)
    # close the pdf document
    pdf.close()

    # save the data into a .dat file
    with open('gauss_params.dat','w') as fw:
        print >> fw, '# Parameters for the Gaussian fits of the all lines'
        print >> fw, '# X_min, X_max, y0, x0, Amp, sigma, fwhm'
        for i in xrange(len(A)):
            print >> fw, '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'.format(min_x[i],max_x[i],y0[i],x0[i],A[i],sigma[i],fwhm[i])


# the function that finds the lines
def find_lines(x_data, y_data, dx, lineheight_min):
	'''
	input is a list of lenght = 2
	input[0] = dx, input[1] = lineheight_min
	march down x array, finding chucks in dx,
	if dy is more than lineheight_min, then
	keep the chuck, then in that chuck, if the
	max of chunk_y is at the center of chunk_x,
	then this will be considered where a line is.
	
	return a list of tuples, (x_min, x_max)
	return an integer representing the number of lines found
	retrun the midpoints of the lines
	
	all is returned in a dictionary.

	limits of where the lines are.
	'''
	ranges = []
	mid_points = []
	n_lines = 0

	for i in xrange(int(len(x_data)-dx+1)):
		# define the chunk to look at
		chunk_y = y_data[i:i+dx]
		
		# see if chunk meets critera
		if meets_critera(chunk_y, lineheight_min):
			ranges.append((x_data[i], x_data[i+dx]))	
			mid_points.append((x_data[i] + x_data[i+dx])/2.0)
			n_lines += 1

	out = {'ranges':ranges, 'n_lines':n_lines, 'mid_points':mid_points}
	return out

# function used my find_lines to see if a line is there
def meets_critera(chunk_y, lineheight_min):
	'''
	returns True if this meets the critera
	that we want to see if we have a line.
	'''
	dy = max(chunk_y) - min(chunk_y)
	loc_max_y = np.where(chunk_y==max(chunk_y))[0][0]
	if (dy >= lineheight_min) and (loc_max_y==np.floor(len(chunk_y)) / 2.0):
		return True
	else:
		return False

"""
# The user will input how many lines they want to find
# and this function optimizes dx and line_height to find
# that number of lines
def opto_params():
	'''
	Use the input box for input variable n_lines.
	return dx, min_line_height

	AS OF NOW, THIS FUNCTION DOES NOT WORK
	'''
	# get the number of lines desired from user
	global xdata
	global ydata
	global n_lines_input
	n_lines = int(n_lines_input.get())

	# do the fit. getting close to n_lines with find_lines
	fit_func = lambda p, x_data, y_data: find_lines(x_data, y_data, p[0], p[1])['n_lines']
	err_func = lambda p, x_data, y_data, y: (fit_func(p, x_data, y_data) - y)
	

	initial_guess = [10, (max(ydata)-min(ydata))/5.0]

	params, covar = leastsq(err_func, initial_guess, args=(xdata, ydata, n_lines) ,full_output=1)
	return params
"""

def plot_it():
	'''
	Plot the entire spectrum with the vlines on it
	'''
	# get the lines, first use opto_params() to set dx and min_line_height
	global xdata
	global ydata
	global dx_input
	global min_line_height

	dx = float(dx_input.get())
	min_line_height = float(min_line_height_input.get())

	v = find_lines(xdata, ydata, dx, min_line_height)['mid_points']

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(xdata, ydata)
	ax.set_ylim(min(ydata), max(ydata))
	if len(v) != 0:
		ax.vlines(v, min(ydata), max(ydata), alpha=0.6)
	fig.show()

# gaussian fitting using non-linear least square fitting 
# with use of scipy.optimize.leastsq

# I will fit data
# and I will plot the data with fits
# if the keyword make_plot is set
# to True.

def gauss_fit(x_data, y_data, make_plot=False, verbose=False):
    """
    doc string yet to be written.
    """
    '''
    # remake the distribution
    d = np.array([np.ones(y_data[i])*x_data[i] for i in xrange(len(x_data))])
    distribution = np.array([np.concatenate([i for i in d])])
    '''
    n_elements = len(x_data)

    # set some intitial guesses for the fit
    y0_0 = min(y_data)
    amp_0 = max(y_data) - min(y_data)
    x0 = x_data[np.floor(n_elements/2.0)]
    width = n_elements/2.5
    initial_guess = [y0_0, amp_0, x0, width]

    gauss = lambda p, x: p[0] + p[1]*np.exp(-(x-p[2])**2 / p[3]**2 )
    err_func = lambda p, x, y: (gauss(p,x)-y)

    coeffs, covar = leastsq(err_func, initial_guess, args=(x_data, y_data))
    if verbose:
		print("y0 = {0}\n".format(coeffs[0]) + 
			  "A = {0}\n".format(coeffs[1]) + 
			  "x0 = {0}\n".format(coeffs[2]) + 
			  "width = {0}".format(coeffs[3]))

    if make_plot:
		x_data2 = np.linspace(min(x_data),max(x_data),num=150)
		def best_fit(x):
			return coeffs[0] + coeffs[1]*np.exp(-(x-coeffs[2])**2 / coeffs[3]**2)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(x_data, y_data, c='b',label='Raw data')
		ax.plot(x_data2, best_fit(x_data2), c='r',label='Best fit')
		ax.set_xlabel("X-Position")
		ax.set_ylabel("Counts")
		ax.set_title("y0={0}\tA={1}\nx0={2}\tsigma={3}\tFWHM={4}".format(coeffs[0],coeffs[1],coeffs[2],coeffs[3],2*np.sqrt(2*np.log(2))*coeffs[3]))
		ax.legend(loc=0)

		return coeffs, fig
    else:
		return coeffs

def load_data():
    '''Open up a menu to click on a file to load
    in for the x and y data of a spectrum.

    Returns two arrays: x and y.  '''

    global xdata
    global ydata

    fname = tkFileDialog.askopenfilename()
    try:
        xdata, ydata = np.loadtxt(fname, unpack=True)
    except:
	    tkMessageBox.showwarning(
		    "Open file error",
		    "Cannot open this file\nMake sure data file is formated correctly.\nSee Help>>Usage"
		    )

### THE MENU BAR ###


def about():
	info = """
Hormiga Spectral Anaysis
A program to analyze spectra. 

This program will find the spectral lines, fit them to Gaussians, and then save the Gaussian parameters to an output file.

Author: Anthony A. Paredes
LLNL, UC Berkeley

Created July 2012
	"""
	tkMessageBox.showinfo("About", info)

def usage():
	usage = """
From the file menu, drop down the menu and click on Load Data.

This will open a window where you can select a text file with the spectral data.

The text data file should be formatted so that there are two columns, the first column being the x-axis and the second colum being the y-axis of the plotted spectrum.

After the data is loaded, you will be returned to the main window.  In this window, you can select the parameters "dx" and "minimum line height".

dx is the width you wish to look for specral lines in.  min_height is the minimum hieght requred for the program to consider a spectral line to be present.

Then click on "Make plot". The plot of the spectrum with vertical lines will appear. In this plot, the vertical lines are where the program belives there are spectral lines.  Fine tune dx and min_height to optimize the line finder.

After you are satisfied with the lines that have been found, click on "Save Gaussian Fits". This will make a pdf with the lines and best fits, as well as a text data file that contains the gaussian parameters.
	"""
	tkMessageBox.showinfo("Usage", usage)

def quit():
    sys.exit(0)

def make_menu(master):
	menubar = Tkinter.Menu(master)

	# create a file menu
	filemenu = Tkinter.Menu(menubar)
	menubar.add_cascade(label="File", menu=filemenu)
	filemenu.add_command(label="Load Data", command=load_data)
	filemenu.add_separator()
	filemenu.add_command(label="Exit", command=quit)

	# create a help menu
	helpmenu = Tkinter.Menu(menubar)
	menubar.add_cascade(label="Help", menu=helpmenu)
	helpmenu.add_command(label="About", command=about)
	helpmenu.add_command(label="Usage", command=usage)
	
	return menubar


########## Here is it running
def main():
	root = Tkinter.Tk()
	root.grid()
	root.title("Hormiga Spectral Analysis")
	root.config(menu=make_menu(root))

	# creat inputs
	global n_lines_input	

	global dx_input
	dx_label = Tkinter.Label(root, text="dx:")
	dx_label.grid(row=1, column=0)
	dx_input = Tkinter.Entry(root)
	dx_input.grid(row=2, column=0)

	global min_line_height_input
	min_line_height_label = Tkinter.Label(root, text="Minimun line height:")
	min_line_height_label.grid(row=1, column=1)
	min_line_height_input = Tkinter.Entry(root)
	min_line_height_input.grid(row=2, column=1)

	# create buttons
	data = Tkinter.Button(root, text="Load Data File", command=load_data)
	data.grid(row=4, columnspan=2)

	make_plot = Tkinter.Button(root, text="Make Plot", command=plot_it)
	make_plot.grid(row=5,columnspan=2)

	make_fits = Tkinter.Button(root, text="Save Gaussian Fits", command=save_gauss_fits)
	make_fits.grid(row=6, columnspan=2)

	root.mainloop()

if __name__=="__main__":
	main()



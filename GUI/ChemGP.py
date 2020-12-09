import Tkinter as tk
import ttk
from tkFileDialog import askopenfilename
import numpy as np

def doNothing():
	print("do nothing")

def import_csv_data():
	global v
	global data
	csv_file_path = tk.askopenfilename()
	v=(csv_file_path)
	sbar_text.delete(1.0,2.0) #clears 'no data'
	sbar_text.insert(END, v+'\n')
	data = np.genfromtxt(csv_file_path, delimiter=",", dtype=None)
	sbar_text.insert(END, "There are "+str(data.shape[0])+" compounds and "+str(data.shape[1])+" columns.\n" )
	sbar_text.insert(END, "Print example datum "+str(data[1][1]))
	
#	sbar_text.insert(END, v+'\n')	
	
	
	# Table for data
	datatree = ttk.Treeview(window)
	datatree["columns"]=("one","two")
	datatree.column("one", width=100)
	datatree.column("two", width=100)
	datatree.heading("one", text="Activity")
	datatree.heading("two", text="column B")
	for i in xrange(0,data.shape[0]-1):
	  datatree.insert("" , i,    text=i+1, values=(str(data[i+1][0]),"1b"))
	datatree.pack()
	
	# Table for images
	imagetree = ttk.Treeview(window)
	imagetree["columns"]=("one","two")
	imagetree.column("one", width=100)
	imagetree.column("two", width=100)
	imagetree.heading("one", text="Image 1")
	imagetree.heading("two", text="Image 2")
	imagetree.insert("" , 0,    text="SMILES", values=("image","image"))
	imagetree.pack()


def set_model():
    pass
    
def help_docs():
    pass


# Main Menu

window = tk.Tk() 

# Program icon
window.title("ChemGP")
img = tk.Image("photo", file="aspirin.png")
window.call('wm','iconphoto',window._w,img)
window.config()

menubar = tk.Menu(window)

# Commands: import a spreadsheet, import last spreadsheet
fileMenu = tk.Menu(menubar, tearoff=0)
fileMenu.add_command(label="Load Data", command=import_csv_data)
fileMenu.add_separator()
fileMenu.add_command(label="Exit", command=window.quit)
menubar.add_cascade(label="File", menu=fileMenu)

modelMenu = tk.Menu(menubar, tearoff=0)
modelMenu.add_command(label="Set Model", command=doNothing)
menubar.add_cascade(label="Model", menu=modelMenu)

helpMenu = tk.Menu(menubar, tearoff=0)
helpMenu.add_command(label="View Help Docs", command=help_docs)
menubar.add_cascade(label="Help", menu=helpMenu)

window.config(menu=menubar)
# Sidebar: Show number of total, input, output, (optimisation) compounds
#	Optional: show model details
sidebar = tk.Frame(window, width=100, bg='white', height=500, relief='sunken', borderwidth=2)
sidebar.pack(expand=False, fill=None, side='left', anchor='nw')
sbar_text = tk.Text(sidebar, height=2)
sbar_text.pack()

sbar_text.insert(END, "no data")



# main content area
mainarea = tk.Frame(window, bg='#CCC', width=400, height=500)
mainarea.pack(expand=False, fill='both')

# Data
#	Plot an input against an output
#	Plot histogram of outputs
#	View compounds
#	Plot similarities
#	Show enantiomers
#	Plot real and predicted outputs
# Model
#	In-built models
#	1D model
#		Plot prior (optional: include random draws)
#		Plot posterior (optional: include random draws)
#	Custom models
#		Hyperparameter selection: Grid search, Latin hypercube, Maximum likelihood
#		Fingerprint selection
#		Fingerprint similarity
#		RDKit descriptor selection
#	
# Regression
#	Other methods of regression
# Bayesian Optimisation
#	Single output
#	Multiple output
# Classification
#	Set threshold
#	View distribution and pick a threshold
#	Plot
#	Enrichment factors
# DFT
# Docs


#canvas1 = Canvas(window, width=200, height=100, bg="#ff8f77")
#canvas1.pack()


#status = Label(window, text="Status..", bd=1, relief=SUNKEN, anchor=W)
#status.pack(side=BOTTOM, fill=X)

window.mainloop()

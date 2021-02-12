__precompile__() # this module is safe to precompile
module HemiVisualise

export 
	NeuronType,
	InputROI,
	OutputROI,
	BodyROI,
	StringID,
	IntID,
	ArrayID,
	get_neurons,
	criterion,
	get_ids,
	get_skeletons,
	upstream_neurons,
	downstream_neurons,
	plot_neurons, plot_neurons!,
	plot_upstream, plot_upstream!,
	plot_downstream, plot_downstream!,
	spinning_camera,
	save_spinning_camera

using CSV, DataFrames, PyCall, AbstractPlotting, Colors, Makie, Infiltrator

const NEU = PyNULL()

function __init__()

	py"""
	import sys
	sys.path.insert(0, "~/.julia/dev/HemiVisualise/src/")
	"""	
	
	pushfirst!(PyVector(pyimport("sys")."path"), "")
	#push!(pyimport("sys")."path", "~/.julia/dev/src/NeuprintInterface.py")
	copy!(NEU, pyimport("NeuprintInterface"))
end

include("FetchNeurons.jl")
include("PlotNeurons.jl")

end

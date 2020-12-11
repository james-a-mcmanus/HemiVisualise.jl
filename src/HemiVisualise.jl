__precompile__() # this module is safe to precompile
module HemiVisualise

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

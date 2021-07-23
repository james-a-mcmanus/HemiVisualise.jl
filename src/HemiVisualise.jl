__precompile__() # this module is safe to precompile
module HemiVisualise

export 
	CellType, InputROI, OutputROI, BodyROI, StringID, IntID, ArrayID,
	neuprint,
	neuron_ids, criterion, get_skeletons, upstream_neurons, downstream_neurons,
	plot_neurons, plot_neurons!, plot_upstream, plot_upstream!, plot_downstream, plot_downstream!,
	spinning_camera, save_spinning_camera


using CSV, DataFrames, PyCall, Colors, Makie, GLMakie, JSON, Crayons

neuprint() = pyimport("neuprint")

function welcome()
	println("Welcome to HemiVisualise!")
	print_hemibrain()
	token, was_saved = get_token()
	set_client(token)
	!was_saved && save_creds(token)
end

function print_hemibrain()
	text = "                                        \n          &&&&&&&&&. &&&&#####          \n        &&&&&&&&&&&&&&&&&#######        \n   ####  &&&&&&&&&&&&&&&& ######  ##    \n #######&&&&&&&&&&&&&&&&& ############# \n #######&&&&&&    &&& &&&###############\n #######&&&&&&&&&&&&# ##################\n ########(###*&# &&######### ###########\n   ########      #########     #########\n      ##                         #####  \n"
	for ch in text
		cl = ch == '&' ? Crayon(foreground=:light_cyan, bold=true) : Crayon(foreground=:white)
		print(cl, ch)
	end
end

function set_client(neuprint, token::AbstractString)
	return neuprint.Client("neuprint.janelia.org", dataset="hemibrain:v1.1", token=token)
end
set_client(token::AbstractString) = set_client(neuprint(), token)

function get_token()
	creds = joinpath(dirname(pathof(HemiVisualise)), "credentials.json")
	has_creds = isfile(creds)
	if has_creds
		tok = JSON.parsefile(creds)["TOKEN"]
	else
		println("\n\nSet up a TOKEN?\nToken: ")
		tok = readline()
	end
	return tok, has_creds
end

include("FetchNeurons.jl")
include("PlotNeurons.jl")

function __init__()
	welcome()
end
end


"""
## TODO ## 
- [x] Save credentials on startup
- [ ] Save ids (skeletons done)
- [ ] More full intuitive criteria 
"""
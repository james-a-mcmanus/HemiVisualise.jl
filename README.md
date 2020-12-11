# HemiVisualise.jl

## Install
Navigate into your .julia/dev directory
```
git clone https://github.com/james-a-mcmanus/HemiVisualise.jl.git
cd HemiVisualise.jl/src
julia
dev HemiVisualise
```

## QuickStart
```
using HemiVisualise # This may take some time
neurons = get_ids(get_neurons(NeuronType("KCg-d")))
skeletons = fetch_skeletons(neurons)
scene = plot_neurons(skeletons, color=RGBA(0,0,0,.2))
```



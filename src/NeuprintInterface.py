from neuprint import Client, NeuronCriteria, fetch_neurons, fetch_adjacencies, merge_neuron_properties, fetch_synapse_connections
import pandas as pd
import numpy as np
import json

with open("credentials.json") as f:
	credentials = json.load(f)


TOKEN = credentials["TOKEN"]
CLIENT = Client('neuprint.janelia.org', dataset='hemibrain:v1.1', token=TOKEN)


def get_skeletons(bodyIds):
	skeletons = []
	
	exceptions = []

	for i, bodyId in enumerate(bodyIds):
		print(i)
		try:
			s = c.fetch_skeleton(bodyId, format='pandas')
			s['bodyId'] = bodyId
			#s['color'] = bokeh.palettes.Accent[5][i]
			skeletons.append(s)
		except:
			exceptions.append(bodyId)


	# Combine into one big table for convenient processing
	#skeletons = pd.concat(skeletons, ignore_index=True)
	#skeletons.head()
	#print(exceptions)
	return skeleton


def get_upstream(bodyIds):
	neuron_df, conn_df = fetch_adjacencies(None, bodyIds)
	conn_df = merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])
	conn_df.replace(to_replace=[None], value="Unknown", inplace=True)
	return conn_df


def get_downstream(bodyIds):
	neuron_df, conn_df = fetch_adjacencies(bodyIds, None)
	conn_df = merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])
	conn_df.replace(to_replace=[None], value="Unknown", inplace=True)
	return conn_df

def synapse_connections(bodyIds):
	eb_conns = fetch_synapse_connections(bodyIds) # try NC(bodyId=bodyIds) if that doesn't work
	return eb_conns


## TODO: Save all of these as csvs so we don't have to worry about using very laggy PyCall.

from neuprint import Client, NeuronCriteria, fetch_neurons
import pandas as pd
import numpy as np

TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImphbWNtYW51czFAc2hlZmZpZWxkLmFjLnVrIiwibGV2ZWwiOiJub2F1dGgiLCJpbWFnZS11cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vLV9VbXJGaVZlRXZVL0FBQUFBQUFBQUFJL0FBQUFBQUFBQUFBL0FNWnV1Y21mSWU5V3ZScm9uQy1ZUUU0ZGRrOFhWZklUdGcvcGhvdG8uanBnP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzcyNDgyNjIwfQ.18JDKwt_yLehRIESw-2PZHbt6Ml_7xiUH-6d6EgsY1E'
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
	return skeletons
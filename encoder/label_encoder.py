import numpy as np


class LabelEncoder:
	"""
	Label Encoder class and encodes categorical labels using norminal encoding
	"""
	
	def __init__(self, label):
		self.label = label
		self.n_classes = len(np.unique(self.label))

	@property
	def transform(self):
		if len(self.label) != 0:
			labels = dict(zip(np.unique(self.label), np.arange(self.n_classes)))
			targets = np.vectorize(labels.get)(self.label)
			return np.asarray(targets)
		else:
			return np.asarray([])
	
	def inverse_transform(self):
		return np.asarray(self.label)
	
	@property
	def classes_(self):
		return np.asarray(np.unique(self.label))
	
	# Alias
	fit_transform = transform

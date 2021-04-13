import numpy as np


class LabelEncoder:
	"""
	Label Encoder class and encodes categorical labels using norminal encoding
	"""
	
	def __init__(self, label):
		self.label = label
		self.n_classes = len(np.unique(self.label))
	
	def fit(self):
		print('LabelEncoder()')
	
	@property
	def transform(self):
		labels = dict(zip(np.unique(self.label), np.arange(self.n_classes)))
		targets = np.vectorize(labels.get)(self.label)
		return targets
	
	@property
	def inverse_transform(self):
		return np.array(self.label)
	
	@property
	def classes_(self):
		# return np.array(sorted([l for i, l in enumerate(self.label) if l not in self.label[:i]]))
		return np.unique(self.label)
	
	def one_hot_encode(self):
		if isinstance(self.label, list):
			y = self.label
		else:
			y = self.label.flatten()
		n_classes = len(np.unique(y))
		labels = dict(zip(np.unique(y), np.arange(n_classes)))
		targets = np.vectorize(labels.get)(y)
		return np.eye(n_classes)[targets]

	# Alias
	fit_transform = transform

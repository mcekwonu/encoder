import numpy as np


class OneHotEncoder:
	"""
	One hot encoder class and encodes categorical labels
	"""
	def __init__(self, label):
		self.label = label
		self.n_classes = len(np.unique(self.label))
	
	@property
	def transform(self):
		"""
		Fits and transforms the input labels into one hot encoding
		"""
		if isinstance(self.label, list):
			y = self.label
		else:
			y = self.label.flatten()

		n_classes = len(np.unique(y))
		if len(y) != 0:
			labels = dict(zip(np.unique(y), np.arange(n_classes)))
			targets = np.vectorize(labels.get)(y)
			return np.eye(n_classes)[targets]
		else:
			return np.asarray([])
		
	@property
	def inverse_transform(self):
		"""
		Returns the input labels
		"""
		return np.array(self.label)
	
	@property
	def classes_(self):
		"""
		Sorts and returns the class labels
		"""
		return np.unique(self.label)
		
	# Alias
	fit_transform = transform

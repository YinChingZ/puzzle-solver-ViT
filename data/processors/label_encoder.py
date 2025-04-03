class LabelEncoder:
    def __init__(self):
        self.label_to_index = {}
        self.index_to_label = {}

    def fit(self, labels):
        unique_labels = set(labels)
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def transform(self, labels):
        return [self.label_to_index[label] for label in labels]

    def inverse_transform(self, indices):
        return [self.index_to_label[idx] for idx in indices]

    def fit_transform(self, labels):
        self.fit(labels)
        return self.transform(labels)

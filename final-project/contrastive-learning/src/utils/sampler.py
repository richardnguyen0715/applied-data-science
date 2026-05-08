import random
from collections import defaultdict
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes, n_samples):
        """
        Args:
            labels: list or tensor of labels
            n_classes: number of classes per batch
            n_samples: number of samples per class
        """
        self.labels = list(labels)

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        # Shuffle indices for each class
        for label in self.label_to_indices:
            random.shuffle(self.label_to_indices[label])

        self.used_label_indices_count = {label: 0 for label in self.label_to_indices}

        self.labels_set = list(self.label_to_indices.keys())

        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_classes * self.n_samples

    def __iter__(self):
        count = 0
        for _ in range(len(self)):
            # Choose random n_classes
            classes = random.sample(self.labels_set, self.n_classes)

            indices = []
            for class_ in classes:
                start = self.used_label_indices_count[class_]
                end = start + self.n_samples

                # If not enough samples, reshuffle
                if end > len(self.label_to_indices[class_]):
                    random.shuffle(self.label_to_indices[class_])
                    start = 0
                    end = self.n_samples

                indices.extend(self.label_to_indices[class_][start:end])
                self.used_label_indices_count[class_] = end

            yield indices
            count += 1

    def __len__(self):
        return len(self.labels) // self.batch_size
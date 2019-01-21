from algorithms.kerasrandominitcnn import KerasRandomInitCNN
from algorithms.keraswtacrnn import KerasWTACRNN


class KerasRandomInitCRNN(KerasRandomInitCNN, KerasWTACRNN):
    def __init__(self, results_dir, config):
        super(KerasRandomInitCRNN, self).__init__(results_dir, config)


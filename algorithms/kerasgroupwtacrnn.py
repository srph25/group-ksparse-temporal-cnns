from algorithms.kerasgroupwtacnn import KerasGroupWTACNN
from algorithms.keraswtacrnn import KerasWTACRNN


class KerasGroupWTACRNN(KerasGroupWTACNN, KerasWTACRNN):
    def __init__(self, results_dir, config):
        super(KerasGroupWTACRNN, self).__init__(results_dir, config)


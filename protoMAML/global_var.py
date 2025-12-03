#Tuning parameters:
'''This is just a class used for looking and editing at all the various parameters used throught the code'''
#Tuning parameters:
'''This is just a class used for looking and editing at all the various parameters used throught the code'''

class TuningParameters():
    #data params
    _train_ways=7                   #Support:Classes
    _train_samples=5                #Support:No.of examples in each class
    _test_samples=15                #Query:Samples
    
    #meta params
    _inner_loop=15                   #how many times the collective losses are summed and updated
    _outer_loop=5000                   #how many distinct losses are calculated  

    #misc params
    _adp_steps=15
    _embedding_size=512

    @property
    def train_ways(self):
        return self._train_ways
    @property
    def train_samples(self):
        return self._train_samples
    @property
    def test_samples(self):
        return self._test_samples
    @property
    def inner_loop(self):
        return self._inner_loop
    @property
    def outer_loop(self):
        return self._outer_loop
    @property
    def adp_steps(self):
        return self._adp_steps
    @property
    def embedding_size(self):
        return self._embedding_size
from DPMERFGenerator import DPMERFGenerator
from TrainRoutine import AutoEncTrainRoutine

class AEDPMERF:
    def __init__(self, seq_len=180, n_feat=1, emb_dim=32, training_data_path="data/normal_train_180.csv",
                 is_priv=False, 
                 n_feat=2000, 
                 input_size=10):
        self.ae = AutoEncTrainRoutine(n_feat=n_feat, emb_dim=emb_dim, training_data_path=training_data_path)
        self.dpmerfgen = DPMERFGenerator()
        
    def save_ae(self, name):
        ae.save_model()
    
    def load_ae(self, name):
        self.ae.load_model()
    
    def train_ae(self, n_epochs=20, lr=5e-4, batch_size=1):
        return self.ae.train_model()
    
    def train_gen(self):
        self.dpmerfgen.train_generator()

    def generate(self):
        gen_enc = self.dpmerfgen.generate()
        gen_dec = self.ae.decode_data()
        return gen_dec
        
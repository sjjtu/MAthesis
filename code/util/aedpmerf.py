from util.DPMERFGenerator import DPMERFGenerator
from util.TrainRoutine import AutoEncTrainRoutine

class AEDPMERF:
    def __init__(self, seq_len=180, n_feat=1, emb_dim=32,
                 is_priv=False, 
                 n_feat_rff=2000, 
                 input_size=10):
        self.ae = AutoEncTrainRoutine(n_feat=n_feat, emb_dim=emb_dim)

        self.dpmerfgen = DPMERFGenerator(is_priv=is_priv, 
                 n_feat=n_feat_rff, 
                 input_size=input_size)
        
    def save_ae(self, name):
        self.ae.save_model(name)
    
    def load_ae(self, name):
        self.ae.load_model(name)
        return self.ae.model
    
    def train_ae(self, train_ds_path, val_ds_path, n_epochs=20, lr=5e-4, batch_size=1):
        return self.ae.train_model(train_ds_path=train_ds_path, val_ds_path=val_ds_path, n_epochs=n_epochs, lr=lr, batch_size=batch_size)
    
    def encode_train_data(self, train_ds_path, fname):
        return self.ae.encode_train_data(train_ds_path=train_ds_path, fname=fname)
    
    def train_gen(self, data, mini_batch_size=0.1, n_epochs=2000, lr=1e-2, eps=1, delt=1e-5):
        self.dpmerfgen.train_generator(data=data, mini_batch_size=mini_batch_size, n_epochs=n_epochs, lr=lr, eps=eps, delt=delt)

    def generate(self, n_gen_samples, fname):
        gen_enc = self.dpmerfgen.generate(n_gen_samples, fname=fname)
        gen_dec = self.ae.decode_data(gen_enc)
        return gen_dec
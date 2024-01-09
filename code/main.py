from util.Preprocessor import Preprocessor
from util.TrainRoutine import AutoEncTrainRoutine
from util.DPMERFGenerator import DPMERFGenerator

#pp = Preprocessor()
#vanilla_ae = AutoEncTrainRoutine()
#vanilla_ae.load_model("lstmae_180_embed32.pth")
#vanilla_ae.train_model()

dpmerfgen = DPMERFGenerator(is_priv=True)
dpmerfgen.train_generator()
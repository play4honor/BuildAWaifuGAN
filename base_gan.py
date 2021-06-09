import torch
import torch.nn

class WassersteinLoss():

    def __init__(self):
        pass

    def __call__(self, y):
        return y[:, 0].sum()

    def gradientPenalty(self):
        pass


class ModelConfig():
    def __init__(self):
        self.dimLatentVector = None
        self.baseLearningRate = None
        self.calculateLoss = None


class BaseGAN():

    """GAN base class"""

    def __init__(
        self,
        dimLatentVector: int,
        baseLearningRate: float,
        device: str,
    ):
        if device not in ['cuda:0', 'cpu']:
            raise ValueError(f"Device must be either cuda:0 or cpu, not whatever nonsense {device} is.")
        
        self.config = ModelConfig()

        self.device = device
        self.config.dimLatentVector = dimLatentVector
        self.config.baseLearningRate = baseLearningRate

        self.generator = None
        self.gen_optimizer = None

        self.discriminator = None
        self.dis_optimizer = None

    def setGen(self, net, optimizer):
        self.generator = net.to(device)
        self.gen_optimizer = optimizer
        self.gen_optimizer.zero_grad()

    def setDis(self, net, optimizer):
        self.discriminator = net.to(device) 
        self.dis_optimizer = optimizer
        self.dis_optimizer.zero_grad() 

    def setLoss(self, lossClass):
        self.config.calculateLoss = lossClass

    def calculateLoss(self, *args):
        return self.config.calculateLoss(*args)

    def _createNoise(self, batchSize):
        noise = torch.randn(batchSize, self.dimLatentVector)
        return noise

    def trainGen(self, batchSize):
        
        if not self._hasGenAndDis():
            raise AttributeError("Training requires both a discriminator and generator to be set.")
        if not self._hasOptimizers():
            raise AttributeError("Training requires optimizers to be set.")

        self.dis_optimizer.zero_grad()
        self.gen_optimizer.zero_grad()

        inputNoise = self._createNoise(batchSize)

        fakeOut = self.generator(inputNoise)
        fakePred = self.discriminator(fakeOut)
        fakeLoss = -self.calculateLoss(fakePred)

        # TKTK maybe there's some gradient penalty shit here?

        fakeLoss.backward()
        self.gen_optimizer.step()

        return fakeLoss.item()


    def trainDis(self, x):
        if not self._hasGenAndDis():
            raise AttributeError("Training requires both a discriminator and generator to be set.")
        if not self._hasOptimizers():
            raise AttributerErrors("Training requires optimizers to be set.")
        
        self.dis_optimizer.zero_grad()
        self.gen_optimizer.zero_grad()
        
        # Real
        realPred = self.discriminator(x)
        realLoss = -self.calculateLoss(realPred)
        
        # Fake
        inputNoise = self._createNoise(x.size()[0])
        fakePred = self.discriminator(inputNoise)
        fakeLoss = self.calculateLoss(fakePred)

        # TKTK maybe there's some gradient penalty shit here?

        totalLoss = realLoss + fakeLoss

        totalLoss.backward()
        self.dis_optimizer.step()

        return totalLoss.item()

    def _hasGenAndDis(self):
        return self.generator is not None and self.discriminator is not None

    def _hasOptimizers(self):
        return self.gen_optimizer is not None and self.dis_optimizer is not None

    def save(self):

        pass

    @staticmethod
    def load():

        pass

if __name__ == "__main__":

    a = BaseGAN(10, 5, "cpu")

    a.setLoss(WassersteinLoss())

    l = torch.randn(32, 1)
    print(a.calculateLoss(l))
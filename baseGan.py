import torch
import torch.nn as nn

class WassersteinLoss():

    def __init__(self):
        self.transform = torch.nn.Sigmoid()

    def __call__(self, y):
        return self.transform(y[:, 0]).mean()

    def gradientPenalty(self, real, generated, dis_net, ):
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
        gradientPenalty=True
    ):
        if device not in ['cuda:0', 'cpu']:
            raise ValueError(f"Device must be either cuda:0 or cpu, not whatever nonsense {device} is.")
        
        self.config = ModelConfig()

        self.device = device
        self.config.dimLatentVector = dimLatentVector
        self.config.baseLearningRate = baseLearningRate
        self.config.gradientPenalty = gradientPenalty

        self.generator = None
        self.gen_optimizer = None

        self.discriminator = None
        self.dis_optimizer = None

    def setGen(self, net, optimizer):
        self.generator = net.to(self.device)
        self.gen_optimizer = optimizer
        self.gen_optimizer.zero_grad()

    def setDis(self, net, optimizer):
        self.discriminator = net.to(self.device) 
        self.dis_optimizer = optimizer
        self.dis_optimizer.zero_grad() 

    def setOptimizers(self, gen_optimizer, dis_optimizer):
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer

        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

    def setLoss(self, lossClass):
        self.config.calculateLoss = lossClass

    def calculateLoss(self, *args):
        return self.config.calculateLoss(*args)

    def _createNoise(self, batchSize):
        noise = torch.randn((batchSize, self.config.dimLatentVector))
        return noise

    def trainGen(self, batchSize):
        
        if not self._hasGenAndDis():
            raise AttributeError("Training requires both a discriminator and generator to be set.")
        if not self._hasOptimizers():
            raise AttributeError("Training requires optimizers to be set.")

        self.dis_optimizer.zero_grad()
        self.gen_optimizer.zero_grad()

        inputNoise = self._createNoise(batchSize).to(self.device)

        fakeOut = self.generator(inputNoise)
        fakePred = self.discriminator(fakeOut)
        fakeLoss = -self.calculateLoss(fakePred)

        # TKTK maybe there's some gradient penalty shit here?

        fakeLoss.backward()
        self.gen_optimizer.step()

        return fakeLoss.item(), fakeOut


    def trainDis(self, x):
        if not self._hasGenAndDis():
            raise AttributeError("Training requires both a discriminator and generator to be set.")
        if not self._hasOptimizers():
            raise AttributeError("Training requires optimizers to be set.")
        
        self.dis_optimizer.zero_grad()
        self.gen_optimizer.zero_grad()
        
        # Real
        realPred = self.discriminator(x)
        realLoss = -self.calculateLoss(realPred)
        
        # Fake
        inputNoise = self._createNoise(x.size()[0]).to(self.device)
        fakeX = self.generator(inputNoise)
        fakePred = self.discriminator(fakeX)
        fakeLoss = self.calculateLoss(fakePred)

        totalLoss = realLoss + fakeLoss

        # TKTK maybe there's some gradient penalty shit here?
        if self.config.gradientPenalty:

            # Generate the interpolated observations
            batchSize = x.size()[0]
            epsilon = torch.rand(batchSize, 1)
            epsilon = epsilon.expand(batchSize, int(x.nelement() /
                                                    batchSize)).contiguous().view(
                                                        x.size())
            epsilon = epsilon.to(self.device)

            mixedObs = epsilon * x + (1-epsilon) * fakeX
            #mixedObs.requires_grad = True


            mixedPred = self.discriminator(mixedObs).sum()

            gradients = torch.autograd.grad(
                outputs=mixedPred,
                inputs=mixedObs,
                create_graph=True,
                retain_graph=True
            )

            gradients = gradients[0].view(batchSize, -1)
            gradients = (gradients * gradients).sum(dim=1).sqrt()
            gradientPenalty = ((gradients - 1) ** 2).mean() * 10

            totalLoss += gradientPenalty

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

    import torch.optim as optim

    class RealDiscriminator(nn.Module):

        def __init__(self):
            super(RealDiscriminator, self).__init__()
            self.linear = nn.Linear(128, 128)
            
        def forward(self, x):
            a = torch.rand(x.shape[0], 1).to("cuda:0")
            b = x.sum(dim = [1, 2])

            return a + b

    a = BaseGAN(10, 5, 'cuda:0')

    dis = RealDiscriminator()
    

    genOpti = optim.AdamW(filter(lambda p: p.requires_grad, gen.parameters()))
    disOpti = optim.AdamW(filter(lambda p: p.requires_grad, dis.parameters()))

    a.setLoss(WassersteinLoss())
    a.setGen(gen, genOpti)
    a.setDis(dis, disOpti)

    realBatch = torch.rand(10, 100, 100).to("cuda:0")

    l = a.trainDis(realBatch)

    print(l)

    
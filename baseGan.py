import torch
import torch.nn as nn

import os
import io
import zipfile
import pickle as pkl

class WassersteinLoss():

    def __init__(self, sigmoid = True):

        if sigmoid == True:
            self.transform = torch.nn.Sigmoid()
        else:
            self.transform = nn.Identity()

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
        dimLatentVector: int = None,
        device: str = None,
        gradientPenalty=True,
        config = None,
        fromFile = None
    ):

        if device not in ['cuda:0', 'cpu']:
            raise ValueError(f"Device must be either cuda:0 or cpu, not whatever nonsense {device} is.")

        if fromFile is not None:

            self.config = config
            self.device = device
            self.generator = fromFile["generator"]
            self.discriminator = fromFile["discriminator"]

        else:

            self.config = ModelConfig()

            self.device = device
            self.config.dimLatentVector = dimLatentVector
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
        fakePred, _ = self.discriminator(fakeOut)
        fakeLoss = -self.calculateLoss(fakePred)

        # TKTK maybe there's some gradient penalty shit here?

        fakeLoss.backward()
        self.gen_optimizer.step()

        # for p in self.generator.parameters():

        #     if (p.grad.data.abs() == math.inf).any():
        #         print(f"Found inf in grads for {p.name}")

        #     if (p.grad.data != p.grad.data).any():
        #         print(f"Found nan in grads for {p.name}")

        return fakeLoss.item(), fakeOut


    def trainDis(self, x, alpha=None):
        if not self._hasGenAndDis():
            raise AttributeError("Training requires both a discriminator and generator to be set.")
        if not self._hasOptimizers():
            raise AttributeError("Training requires optimizers to be set.")

        loss_dict = dict()
        
        self.dis_optimizer.zero_grad()
        self.gen_optimizer.zero_grad()
        
        # Real
        realPred, dis_activation = self.discriminator(x)
        realLoss = -self.calculateLoss(realPred)
        
        # Fake
        inputNoise = self._createNoise(x.size()[0]).to(self.device)
        fakeX = self.generator(inputNoise).detach()
        fakePred, _ = self.discriminator(fakeX)
        fakeLoss = self.calculateLoss(fakePred)

        loss_dict["dis_real"] = realLoss
        loss_dict["dis_fake"] = fakeLoss

        totalLoss = (realLoss + fakeLoss) / 2

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
            mixedObs.requires_grad = True

            mixedPreds, _ = self.discriminator(mixedObs)
            mixedPred = mixedPreds.sum()

            gradients = torch.autograd.grad(
                outputs=mixedPred,
                inputs=mixedObs,
                create_graph=True,
                retain_graph=True
            )

            gradPenWeight = 10 if alpha is None else 10 * alpha

            gradients = gradients[0].view(batchSize, -1)
            gradients = (gradients * gradients).sum(dim=1).sqrt()
            # May want to consider scaling this by (1-alpha)
            gradientPenalty = ((gradients - 1) ** 2).mean() * gradPenWeight

            loss_dict["non_grad_loss"] = totalLoss.clone().item()
            loss_dict["grad_loss"] = gradientPenalty.item()
            totalLoss += gradientPenalty
            

        else:
            loss_dict["non_grad_loss"] = totalLoss.item()
            loss_dict["grad_loss"] = 0

        loss_dict["total_loss"] = totalLoss.item()

        totalLoss.backward()
        self.dis_optimizer.step()

        return loss_dict

    def _hasGenAndDis(self):
        return self.generator is not None and self.discriminator is not None

    def _hasOptimizers(self):
        return self.gen_optimizer is not None and self.dis_optimizer is not None

    def save(self, path):

        torch.save(self.generator, "___gen.pt")
        torch.save(self.discriminator, "___dis.pt")

        with open("___config.pkl", mode="wb") as f:
            pkl.dump(self.config, f)

        with zipfile.ZipFile(path, mode="w") as f:
            f.write("___gen.pt", "gen.pt")
            f.write("___dis.pt", "dis.pt")
            f.write("___config.pkl", "config.pkl")

        os.remove("___gen.pt")
        os.remove("___dis.pt")
        os.remove("___config.pkl")

    @staticmethod
    def load(path, device):

        net_dict = {}

        with zipfile.ZipFile(path, mode="r") as f:

            with io.BytesIO(f.read("gen.pt")) as g:
                net_dict["generator"] = torch.load(g, map_location=device)

            with io.BytesIO(f.read("dis.pt")) as d:
                net_dict["discriminator"] = torch.load(d, map_location=device)

            with io.BytesIO(f.read("config.pkl")) as c:
                config = pkl.load(c)

        bg = BaseGAN(device=device, config=config, fromFile=net_dict)

        return bg

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

    
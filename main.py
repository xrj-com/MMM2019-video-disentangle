import torch
import torch.nn as nn
import torchvision
from config import opt
from models import ContentEncoder, PoseEncoder, Decoder, SceneDiscriminator, RecDiscriminator
from torch.utils.data import DataLoader
from LASIESTA_dataloader import scenePairs_dataset, plot_dataset
import utils
import os
if not os.path.exists(opt.save):
        os.makedirs(opt.save)

from dumblog import dlog
trainLogger = dlog(os.path.join(opt.save,'train'))
valLogger = dlog(os.path.join(opt.save,'val'))


device = torch.device("cuda:{:d}".format(opt.gpu) if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, opt):
        self.opt = opt
        self.netCE = ContentEncoder(inD=3, contentDim=opt.contentDim, is_norm=opt.normalize).to(device)
        self.netPE = PoseEncoder(inD=3, poseDim=opt.poseDim, is_norm=opt.normalize).to(device)
        self.netDE = Decoder(inD=opt.contentDim+opt.poseDim).to(device)
        self.netSD = SceneDiscriminator(poseDim=opt.poseDim).to(device)
        # my work
        self.netRD = RecDiscriminator(inD=6).to(device)

        self.netCE.weight_init(mean=0, std=0.02)
        self.netPE.weight_init(mean=0, std=0.02)
        self.netDE.weight_init(mean=0, std=0.02)
        self.netRD.weight_init(mean=0, std=0.02)

        self.optimCE = torch.optim.Adam(self.netCE.parameters(), lr=opt.learningRate, betas=(opt.beta1, 0.999))
        self.optimPE = torch.optim.Adam(self.netPE.parameters(), lr=opt.learningRate, betas=(opt.beta1, 0.999))
        self.optimDE = torch.optim.Adam(self.netDE.parameters(), lr=opt.learningRate, betas=(opt.beta1, 0.999))
        self.optimSD = torch.optim.Adam(self.netSD.parameters(), lr=opt.learningRate, betas=(opt.beta1, 0.999))
        self.optimRD = torch.optim.Adam(self.netRD.parameters(), lr=opt.learningRate, betas=(opt.beta1, 0.999))

        self.trainDataloader = DataLoader(scenePairs_dataset(opt.dataRoot, opt.epochSize, opt.maxStep), opt.batchSize, num_workers=4)
        self.valDataloader = DataLoader(scenePairs_dataset(opt.dataRoot, 10, opt.maxStep), opt.batchSize, num_workers=4)
        self.plotDateloader = DataLoader(plot_dataset(opt.dataVal, 10, 20, delta=4), 20)

        self.rec_criterion = nn.MSELoss()
        self.sim_criterion = nn.MSELoss()
        self.bce_criterion = nn.BCELoss()

    def train_recon_discriminator(self, pred, img2, origin_target, iter=5):
        self.netRD.train()
        pred = pred.detach()
        img2 = img2.detach()
        target_pred = origin_target.detach()
        target_img2 = origin_target.detach()
        target_pred[:] = 0.0
        target_img2[:] = 1.0
        target = torch.cat([target_pred, target_img2], 0)


        input1 = torch.cat([pred, img2], 0)
        input2 = torch.cat([img2, img2], 0)

        for i in range(iter):
            self.optimRD.zero_grad()
            out = self.netRD(input1, input2)
            loss = self.bce_criterion(out, target)
            # print(loss)
            loss.backward()
            self.optimRD.step()    
        #     acc += torch.mean((out > 0.5).to(torch.float32) - target)
        # acc = acc / iter
        self.netRD.eval()
        # return acc.item()

    def train_scene_discriminator(self, img1, img2, origin_target):
        self.optimSD.zero_grad()
        batch_size = opt.batchSize
        half = int(batch_size / 2)
        target = origin_target.clone()
        target[half:] = 0
        hp1 = self.netPE(img1).clone()
        hp2 = self.netPE(img2).clone()
        # ----
        rp = torch.arange(batch_size)
        rp[half:] = torch.randperm(half)+half
        hp2 = hp2[rp.to(torch.long)]
        out = self.netSD(hp1, hp2)
        nll = self.bce_criterion(out, target.view(-1, 1).to(torch.float32))
        nll.backward()
        acc_same = torch.sum(out[:half]>=0.5)
        acc_diff = torch.sum(out[half]<0.5)

        self.optimSD.step()
        return nll.item(), acc_same.item(), acc_diff.item()

    def train(self, img1, img2, img3, origin_target):
        target = origin_target.clone()
        target2 = origin_target.clone()
        self.optimPE.zero_grad()
        self.optimCE.zero_grad()
        self.optimDE.zero_grad()

        hc2 = self.netCE(img2).clone()
        hc1 = self.netCE(img1)
        hp1 = self.netPE(img3).clone()
        hp2 = self.netPE(img2)

        # minimize ||hc1 - hc2||
        latent_mse = torch.mean((hc1 - hc2)**2)
        loss = latent_mse

        # maximize entropy of scene discrimintor output
        nll = 0
        if self.opt.advWeight > 0:
            target[:] = 0.5
            out = self.netSD(hp1, hp2)
            nll = self.bce_criterion(out, target.view(-1, 1).to(torch.float32))
            loss += self.opt.advWeight * nll

        # minimize ||P(hc1, hp2), x2||
        pred = self.netDE(hc1, hp2)
        
        if self.opt.is_advt:
            self.train_recon_discriminator(pred, img2, target.view(-1, 1).to(torch.float32), iter=5)
            pred_score = self.netRD(pred, img2)
            pred_nll = self.bce_criterion(pred_score, target2.view(-1, 1).to(torch.float32))
            loss += pred_nll


        pred_mse = self.rec_criterion(pred, img2)
        loss += pred_mse
        
        loss.backward()

        self.optimCE.step()
        self.optimPE.step()
        self.optimDE.step()

        return pred_mse.item(), latent_mse.item()

    def test(self, img1, img2, img3, origin_target):
        target = origin_target.clone()

        hc2 = self.netCE(img2).clone()
        hc1 = self.netCE(img1)
        hp1 = self.netPE(img3).clone()
        hp2 = self.netPE(img2)

        # minimize ||hc1 - hc2||
        latent_mse = torch.mean((hc1 - hc2)**2)

        # maximize entropy of scene discrimintor output
        target[:] = 0.5
        out = self.netSD(hp1, hp2)
        nll = self.bce_criterion(out, target.view(-1, 1).to(torch.float32))
        acc = torch.sum(out>=0.5)

        # minimize ||P(hc1, hp2), x2||
        pred = self.netDE(hc1, hp2)
        pred_mse = self.rec_criterion(pred, img2)

        return pred_mse.item(), latent_mse.item(), nll.item(), acc.item()

    def save_chkpt(self, is_best):
        utils.save_checkpoint(
            {
                'epoch': self.epoch_now,
                'netCE': self.netCE.state_dict(),
                'netPE': self.netPE.state_dict(),
                'netDE': self.netDE.state_dict(),
                'netSD': self.netSD.state_dict(),
                'netRD': self.netRD.state_dict(),
                'optimCE': self.optimCE.state_dict(),
                'optimPE': self.optimPE.state_dict(),
                'optimDE': self.optimDE.state_dict(),
                'optimSD': self.optimSD.state_dict(),
                'optimRD': self.optimRD.state_dict(),

                'total_iter': self.total_iter,
                'best_rec': self.best_rec,
            }, is_best, 'checkpoint', self.opt.save
        )

    def load_chkpt(self, is_best):
        if is_best:
            filename = 'model_best.pth.tar'
        else:
            filename = 'checkpoint.pth.tar'
        model_path = os.path.join(opt.save, filename)
        try:
            checkpoint = torch.load(model_path)
            self.epoch_now = checkpoint['epoch']
            self.total_iter = checkpoint['total_iter']
            self.best_rec = checkpoint['best_rec']
            self.netCE.load_state_dict(checkpoint['netCE'])
            self.netPE.load_state_dict(checkpoint['netPE'])
            self.netDE.load_state_dict(checkpoint['netDE'])
            self.netSD.load_state_dict(checkpoint['netSD'])
            self.netRD.load_state_dict(checkpoint['netRD'])

            self.optimCE.load_state_dict(checkpoint['optimCE'])
            self.optimPE.load_state_dict(checkpoint['optimPE'])
            self.optimDE.load_state_dict(checkpoint['optimDE'])
            self.optimSD.load_state_dict(checkpoint['optimSD'])
            self.optimRD.load_state_dict(checkpoint['optimRD'])
            print('Success loading {}th epoch checkpoint!'.format(self.epoch_now))
        except:
            print('Failed to load checkpoint!')

    def plot(self, dir_name, is_swap=False, hc_zero = False, hp_zero = False):
        self.netPE.eval()
        self.netCE.eval()
        self.netDE.eval()
        hp_seq = []
        hc = []
        # dir = 'plot_swap' if is_swap else 'plot'
        # dir = "plot_hc_zero"
        save_path = os.path.join(self.opt.save, dir_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, vids in enumerate(self.plotDateloader):
            torchvision.utils.save_image(vids,  os.path.join(save_path, 'origin{}.jpg'.format(i)), nrow=4, normalize=True)
            vids = vids.to(device)
            hp_seq.append(self.netPE(vids).clone())
            hc.append(self.netCE(vids[0:1]))
        max_step = hp_seq[0].size(0)
        sample_num = len(hp_seq)
        for i in range(sample_num):
            if is_swap:
                ii = 0
            else:
                ii = i
            if hc_zero:
                hc[i][:] = 0.0
            if hp_zero:
                hp_seq[ii][:] = 0.0

            pred = self.netDE(hc[i].repeat(max_step,1,1,1), hp_seq[ii])
            torchvision.utils.save_image(pred, os.path.join(save_path, 'pred{}.jpg'.format(i)), nrow=4, normalize=True)
        


    def unit_test(self):
        pred = torch.randn(10, 3, 64, 64)
        img2 = torch.randn(10, 3, 64, 64)
        target = torch.randn(10, 1)
        self.train_recon_discriminator(pred, img2, target)
        

    def evaluation(self, is_best):
        self.netCE.eval()
        self.netPE.eval()
        self.netDE.eval()
        self.netSD.eval()
        print('Evalation..')
        self.load_chkpt(is_best=is_best)
        self.plot('eval_plot')
        self.plot('eval_plot_swap', is_swap=True)
        self.plot('eval_plot_hc_zero', hc_zero=True)
        self.plot('eval_plot_hp_zero', hp_zero=True)
        # self.plot(is_swap=True)

    def run(self, resume=True, is_best=False):
        self.best_rec = 1e10
        self.total_iter = 0
        self.epoch_now = 0

        if resume:
            self.load_chkpt(is_best=is_best)

        for ii in range(self.epoch_now + 1 , opt.nEpochs):
            self.netCE.train()
            self.netPE.train()
            self.netDE.train()
            self.netSD.train()

            iteration, pred_mse, latent_mse, sd_acc, sd_nll = 0, 0, 0, 0, 0
            for img1, img2, img3, target in self.trainDataloader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                img3 = img3.to(device)
                target = target.to(device)
                nll, acc_s, acc_d = self.train_scene_discriminator(img1, img2, target)
                sd_nll += nll
                sd_acc += (acc_s + acc_d)

                p_mse, l_mse = self.train(img1, img2, img3, target)
                pred_mse += p_mse
                latent_mse += l_mse
                iteration += 1
                self.total_iter += 1

            trainLogger.info('{:d}\tprediction mse = {:.4f}, latent mse = {:.4f},'
            ' scene disc acc = {:.4f}%, scene disc nll = {:.4f}'.format(
                self.total_iter, 
                pred_mse/iteration, 
                latent_mse/iteration,
                100*sd_acc/(opt.batchSize*iteration), 
                sd_nll/iteration))

            self.netCE.eval()
            self.netPE.eval()
            self.netDE.eval()
            self.netSD.eval()
 
            iteration, pred_mse, latent_mse, sd_acc, sd_nll = 0, 0, 0, 0, 0

            for img1, img2, img3, target in self.valDataloader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                img3 = img3.to(device)
                target = target.to(device)
                p_mse, l_mse, nll, acc = self.test(img1, img2, img3, target)
                pred_mse += p_mse
                latent_mse += l_mse
                sd_nll += nll
                sd_acc += acc
                iteration += 1
            
            valLogger.info('\tprediction mse = {:.4f}, latent mse = {:.4f}'.format(pred_mse/iteration, latent_mse/iteration))
            self.epoch_now += 1
            if (pred_mse/iteration) < self.best_rec:
                self.best_rec = pred_mse/iteration
                print('Saving best model so far (pred mse = {:.4f}) '.format(pred_mse/iteration) + opt.save + '/model_best.pth.tar')
                self.save_chkpt(is_best=True)
            else:
                self.save_chkpt(is_best=False)

            self.plot('train_plot')
            self.plot('train_plot_swap', is_swap=True)
            self.plot('train_plot_hc_zero', hc_zero=True)
            self.plot('train_plot_hp_zero', hp_zero=True)


if __name__ == "__main__":

    trainer = Trainer(opt)

    if opt.eval:
        trainer.evaluation(False)
    else:
        utils.backup_src('./', os.path.join(opt.save, 'backUpSrc'))
        trainer.run()
        
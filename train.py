import argparse
import os
import time
import random

import numpy as np
import torch

import torch.optim as optim

from tqdm import tqdm

from utils.dataset import get_loader
from model import myModel
from datetime import datetime

from utils.loss_funcs import PerceptualLoss, L1_Charbonnier_loss, SSIMLoss, EdgeAwareLoss
from utils.metrics import Evaluator
from model_repos.CIDNet.CIDNet import CIDNet
import yaml
from pathlib import Path


# 设置随机种子
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Trainer(object):
    def __init__(self,args):
        """
        Initialize the Trainer with arguments from the command line or defaults.
        """
        self.args = args
        self.evaluator = Evaluator()
        self.model = (myModel(in_channels=3, feature_channels=32, use_white_balance=True)
                      .to('cuda'))

        ###----------- HVI网络加载 --------------###
        self.hvi_net = CIDNet().cuda()  # CIDNet是用于HVI颜色空间转换的预训练模型
        pth = r"model_files/CIDNet_weight_LOLv2_bestSSIM.pth"
        self.hvi_net.load_state_dict(torch.load(pth, map_location="cuda", weights_only=True))
        self.hvi_net.eval()
        ###-----------end-----------------------###

        ###-------------创建输出目录, 保存模型-------------------###
        now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        print("the time is: ",now_str)
        self.model_save_path = os.path.join(args.save_path, args.model_name, args.dataset)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        ###----------------------end--------------------------###

        ###-------------------从checkpoint中恢复训练-----------###
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict =  {}
            state_dict = self.model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.model.load_state_dict(state_dict)
        ###------------------------end----------------------------###

        ###--------------优化器--------------------###
        self.optim = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
        ###-----------end--------------------------###

        ###----------------调度器-------------------###
        if args.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optim, args.epoch,
                eta_min=args.lr * 1e-4,
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optim, step_size=args.decay_epoch, gamma=args.decay_rate
            )
        ###----------------end-----------------------------###

        ###--------------------加载数据集---------------------###

        # 读取配置文件
        config_path = Path("configs/datasets.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            dataset_configs = yaml.safe_load(f)

        if args.dataset not in dataset_configs:
            raise ValueError(f"Unknown dataset: {args.dataset}. Available: {list(dataset_configs.keys())}")

        cfg = dataset_configs[args.dataset]

        # 自动把配置写进 args
        args.train_root = cfg["train_root"]
        args.val_root = cfg["val_root"]
        args.datasize = cfg["datasize"]
        args.resize = cfg["resize"]
        ###--------------------------------end-----------------------------------###

        ###---------------------------损失函数-------------------------------###
        self.vggL = PerceptualLoss()
        self.L1L = L1_Charbonnier_loss()
        self.ssimL = SSIMLoss(device=torch.device("cuda"),window_size=5)
        self.edgeL = EdgeAwareLoss(loss_type="l2",device=torch.device("cuda"))
        ###----------------------end-----------------------------------###

    def training(self):
        best_psnr = 0.0
        best_round = []
        torch.cuda.empty_cache()

        train_data_loader = get_loader(
            self.args.train_root,
            self.args.train_batch_size,
            self.args.datasize,
            resize=self.args.resize,
            train=True,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,

        )
        self.model.train()

        for epoch in range(1,self.args.epoch + 1):
            loop = tqdm(
                enumerate(train_data_loader),
                total=len(train_data_loader),
                desc=f"Epoch {epoch}",
                leave=False,
            )
            loss_mean = 0.0
            for _, (x,label,_) in loop:
                x = x.to("cuda")
                label = label.to("cuda")
                pred = self.model(x)
                self.optim.zero_grad()

                with torch.no_grad():
                    label_hvi = self.hvi_net.trans.HVIT(label)
                    pred_hvi = self.hvi_net.trans.HVIT(pred.clamp(0.0, 1.0))

                ###----------计算损失-----------###
                hvi_loss = self.L1L(pred_hvi, label_hvi)
                l1_loss = self.L1L(pred, label)
                vgg_loss = self.vggL(pred, label)
                ssim_loss = self.ssimL(pred, label)
                edge_loss = self.edgeL(pred, label)
                final_loss = (
                        l1_loss
                        + 0.5 * hvi_loss
                        + 0.1 * ssim_loss
                        + 0.1 * vgg_loss
                        + 0.1 * edge_loss
                )
                loss_mean += final_loss.item()  # 累积平均损失
                ### 反向传播 + 更新
                final_loss.backward()
                self.optim.step()

                loop.set_description(f"[{epoch}/{self.args.epoch}]")
                loop.set_postfix(loss=final_loss.item())

            print(
                f"[{epoch}/{self.args.epoch}], avg. loss is {loss_mean / len(train_data_loader)}, learning rate is {self.optim.param_groups[0]['lr']}"
            )

            ###-------------保存最好模型----------------###
            if epoch % self.args.epoch_val == 0:
                self.model.eval()  # 评估模式
                ssim_, psnr_ = self.validation()

                if not os.path.exists(self.model_save_path):
                    os.makedirs(self.model_save_path)
                if psnr_ > best_psnr:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(str(self.model_save_path), f"best_model.pth"),
                    )
                    best_psnr = psnr_
                    best_round = {
                        "best epoch": epoch,
                        "best PSNR": best_psnr,
                        "best SSIM": ssim_,
                    }
                    with open(
                            os.path.join(str(self.model_save_path), "records.txt"), "a"
                    ) as f:
                        str_ = "## best round ##\n"
                        for k, v in best_round.items():
                            str_ += f"{k}: {v}. "
                        str_ += "\n####################################"
                        f.write(str_ + "\n")
                with open(os.path.join(str(self.model_save_path), "records.txt"), "a") as f:
                    str_ = f"[epoch: {epoch}], PSNR: {psnr_}, SSIM: {ssim_}"
                    f.write(str_ + "\n")

                self.model.train()  # 恢复训练
            self.scheduler.step()  # 更新lr

        print("The accuracy of the best round is ", best_round)

     ###------验证val数据集的平均PSNR/SSIM
    def validation(self):
        self.evaluator.reset()  # 重置评估器
        val_data_loader = get_loader(
            self.args.val_root,
            self.args.eval_batch_size,
            self.args.datasize,
            train=False,
            resize=self.args.resize,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
        )
        torch.cuda.empty_cache()

        with torch.no_grad():
            loop = tqdm(
                enumerate(val_data_loader), total=len(val_data_loader), leave=False
            )
            for _, (x, label, _) in loop:
                x = x.to("cuda")
                label = (
                    label.numpy().astype(np.float32).transpose(0, 2, 3, 1)
                )  # B, H, W, C
                pred = self.model(x)
                pred = torch.clamp(pred, 0.0, 1.0)
                pred = (
                    pred.data.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
                )  # B, H, W, C

                self.evaluator.evaluation(pred, label)
                loop.set_description("[Validation]")
        ssim_, psnr_ = self.evaluator.getMean()

        print("[Validation] SSIM: %.4f, PSNR: %.4f" % (ssim_, psnr_))
        return ssim_, psnr_





def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=100, help="epoch number")
    parser.add_argument("--epoch_val", type=int, default=1, help="training batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    parser.add_argument("--train_batch_size", type=int, default=24)
    parser.add_argument("--eval_batch_size", type=int, default=24)
    parser.add_argument(
        "--decay_rate", type=float, default=0.1, help="decay rate of learning rate"
    )  ##
    parser.add_argument(
        "--decay_epoch", type=int, default=50, help="every n epochs decay learning rate"
    )  ##
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", type=str, default="cosine")

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--dataset", type=str, default="UFO", choices=["UIEB", "LSUI", 'UFO', 'EUVP-s', 'EUVP-d'])
    parser.add_argument("--model_name", type=str, default="WWE-UIE")
    parser.add_argument("--save_path", type=str, default="./output/")

    parser.add_argument("--resume", type=str)
    parser.add_argument("--gpu_id", type=str, default="0")

    args = parser.parse_args()

    #指定使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    trainer = Trainer(args)
    trainer.training()
    print(args)





if __name__ == '__main__':
    start_time = time.time()
    seed_everything(42)
    main()
    end_time = time.time()

    print("The total training time is:", end_time - start_time) # 单位 s

import argparse

import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import warmup_scheduler
import numpy as np

from utils import get_model, get_dataset, get_experiment_name, get_criterion
from da import CutMix, MixUp

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", required=True, help="API Key for Comet.ml")

parser.add_argument("--dataset", default="c10", type=str, help="[c10, c100]")
parser.add_argument("--in-c", default=3, type=int)
parser.add_argument("--num-classes", default=10, type=int)
parser.add_argument("--model-name", default="vits", help="[vits]", type=str)
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max-epochs", default=200, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=1e-1, type=float)
parser.add_argument("--warmup-epoch", default=5, type=int)
parser.add_argument("--precision", default=16, type=int)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--dropout", default=0.1, type=float)
args = parser.parse_args()
args.benchmark = True if not args.off_benchmark else False
args.gpus = torch.cuda.device_count()
args.num_workers = 4*args.gpus if args.gpus else 8
if not args.gpus:
    args.precision=32

train_ds, test_ds = get_dataset(args)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)

class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams = hparams
        self.model = get_model(hparams)
        self.criterion = get_criterion(args)
        if hparams.cutmix:
            self.cutmix = CutMix(hparams.size, beta=1.)
        if hparams.mixup:
            self.mixup = MixUp(alpha=1.)
        self.log_image_flag = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.hparams.warmup_epoch, after_scheduler=self.base_scheduler)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        if self.hparams.cutmix or self.hparams.mixup:
            if self.hparams.cutmix:
                img, label, rand_label, lambda_= self.cutmix((img, label))
            elif self.hparams.mixup:
                if np.random.rand() <= 0.8:
                    img, label, rand_label, lambda_ = self.mixup((img, label))
                else:
                    img, label, rand_label, lambda_ = img, label, torch.zeros_like(label), 1.
            if not self.log_image_flag and not self.hparams.dry_run:
                self.log_image_flag = True
                self._log_image(img.clone().detach().cpu())

            out = self.model(img)
            loss = self.criterion(out, label)*lambda_ + self.criterion(out, rand_label)*(1.-lambda_)
        else:
            out = self(img)
            loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss", loss)
        self.log("acc", acc)
        return loss

    def training_epoch_end(self, outputs):
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def _log_image(self, image):
        grid = torchvision.utils.make_grid(image, nrow=4)
        self.logger.experiment.log_image(grid.permute(1,2,0))
        print("[INFO] LOG IMAGE!!!")


if __name__ == "__main__":
    experiment_name = get_experiment_name(args)
    print(experiment_name)
    logger = pl.loggers.CometLogger(
        api_key=args.api_key,
        save_dir=".",
        project_name="VisionTransformer",
        experiment_name=experiment_name
    )
    args.api_key = None # Initialize API Key for privacy.
    net = Net(args)
    trainer = pl.Trainer(precision=args.precision,fast_dev_run=args.dry_run, gpus=args.gpus, benchmark=args.benchmark, logger=logger, max_epochs=args.max_epochs, weights_summary="full", progress_bar_refresh_rate=0)
    trainer.fit(model=net, train_dataloader=train_dl, val_dataloaders=test_dl)
    if not args.dry_run:
        model_path = f"weights/{experiment_name}.pth"
        torch.save(net.state_dict(), model_path)
        logger.experiment.log_asset(file_name=experiment_name, file_data=model_path)

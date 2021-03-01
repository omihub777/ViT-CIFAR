import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils import get_model, get_dataset, get_experiment_name
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--api-key", required=True, help="API Key for Comet.ml")

parser.add_argument("--dataset", default="c10", type=str, help="[c10, c100]")
parser.add_argument("--in-c", default=3, type=int)
parser.add_argument("--num-classes", default=10, type=int)
parser.add_argument("--model-name", default="vit_s", help="[vit_s]", type=str)
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--lr", default=1e-1, type=float)
parser.add_argument("--milestones", default=[100, 150], nargs="+", type=int)
parser.add_argument("--gamma", default=1e-1, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max-epochs", default=200, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=1e-4, type=float)
parser.add_argument("--precision", default=16, type=int)

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
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = self.accuracy(F.softmax(out, dim=1), label)
        self.log("loss", loss)
        self.log("acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = self.accuracy(F.softmax(out, dim=1), label)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

if __name__ == "__main__":
    experiment_name = get_experiment_name(args)
    print(experiment_name)
    logger = pl.loggers.CometLogger(
        api_key=args.api_key,
        save_dir=".",
        project_name="image_classification_pytorch",
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

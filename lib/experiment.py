from copy import deepcopy
from datetime import datetime
from os import makedirs, remove
from os.path import join, isfile, isdir, dirname

import numpy as np
import torch


def append_to_file(file, string):
    dir_nm = dirname(file)
    if len(dir_nm) > 0 and not isdir(dir_nm):
        makedirs(dir_nm)
    with open(file, "a") as f:
        f.write(string + "\n")


def train_model(model, dataloader, device, criterion, optimizer):
    model.train()
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        if out.ndim > 1 and out.shape[1] == 1:
            out = out.squeeze(dim=1)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()


def eval_model(model, dataloader, device):
    y_pred, y_true = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            out = model(xb)
            if out.ndim > 1 and out.shape[1] == 1:
                out.squeeze_(dim=1)
            y_pred.append(out.detach_().cpu())
            y_true.append(yb)
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
    return y_pred, y_true


class FunctionEvaluator:
    def __init__(self,
                 func_lst):
        self.func_lst = func_lst

    def __call__(self, y_pred, y_true):
        res = {}
        for fname, func, kwargs in self.func_lst:
            res[fname] = func(y_pred, y_true, **kwargs)
        return res


class Logger:
    def __init__(self, save_path=None):
        self.save_path = save_path

    def log(self, msg):
        print(msg)
        if self.save_path is not None:
            append_to_file(self.save_path, msg)


class TopNSaver:
    def __init__(self, n):
        self.n = n
        self.dct = {0: None}

    def save(self, score, state, save_path):
        if any(score > key for key in self.dct) and all(score != key for key in self.dct):
            if len(self.dct) >= self.n:
                key_to_delete = sorted(list(self.dct.keys()))[0]
                if self.dct[key_to_delete] is not None:
                    try:
                        remove(self.dct[key_to_delete])
                    except OSError:
                        pass
                self.dct.pop(key_to_delete)
            self.dct[score] = save_path
            torch.save(state, save_path)


class Experiment:
    def __init__(self,
                 dl_train,
                 dl_train_val,
                 dl_validation,
                 model,
                 optimizer,
                 criterion,
                 device,
                 max_epoch,
                 metrics,
                 target_metric,
                 format_str,
                 init_epoch=0,
                 scheduler=None,
                 load_path=None,
                 save_path=None,
                 early_stopping=None,
                 evaluate_freq=1,
                 ):
        self._params = locals()
        self.dl_train = dl_train
        self.dl_train_val = dl_train_val
        self.dl_validation = dl_validation
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.max_epoch = max_epoch
        self.metric_evaluator = FunctionEvaluator(metrics)
        self.target_metric = target_metric
        self.format_str = format_str
        self.init_epoch = init_epoch
        self.scheduler = scheduler
        self.load_path = load_path
        self.save_path = save_path
        self.logger = Logger(join(save_path, "log.txt")) if save_path is not None else Logger()
        self.early_stopping = early_stopping
        self.evaluate_freq = evaluate_freq
        self.top5saver = TopNSaver(10)

        self.reset()

    def reset(self):
        self.results = {
            "metrics_train": [],
            "metrics_valid": [],
            "state_dict": None,
        }

        self.best_validation_metric = .0
        self.model_best_state_dict = None
        self.no_score_improvement = 0
        self.experiment_start = datetime.now()
        self.now = None

    def evaluate(self, epoch, step):
        # evaluate subset of train set (in eval mode)
        y_pred_train, y_true_train = eval_model(model=self.model,
                                                dataloader=self.dl_train_val,
                                                device=self.device)
        metrics_train = self.metric_evaluator(y_pred_train, y_true_train)
        self.results["metrics_train"].append(metrics_train)

        # evaluate validation subset
        y_pred_valid, y_true_valid = eval_model(model=self.model,
                                                dataloader=self.dl_validation,
                                                device=self.device)
        metrics_valid = self.metric_evaluator(y_pred_valid, y_true_valid)
        self.results["metrics_valid"].append(metrics_valid)

        val_score = metrics_valid[self.target_metric]
        # check if validation score is improved
        if val_score > self.best_validation_metric:
            self.model_best_state_dict = deepcopy(self.model.state_dict())
            self.best_validation_metric = val_score
            # reset early stopping counter
            self.no_score_improvement = 0
            # save best model weights
            if self.save_path is not None:
                torch.save(self.model_best_state_dict, join(self.save_path, "best_weights.pth"))
        else:
            self.no_score_improvement += 1
            if self.early_stopping is not None and self.no_score_improvement >= self.early_stopping:
                self.logger.log("Early stopping at epoch %d, step %d" % (epoch, step))
                return True

        if self.scheduler is not None:
            self.scheduler.step(val_score)

        if self.save_path is not None:
            # (optional) save model state dict at end of each epoch
            self.top5saver.save(val_score,
                                self.model.state_dict(),
                                join(self.save_path, "model_state_{}_{}.pth".format(epoch, step)))
            # torch.save(self.model.state_dict(), join(self.save_path, "model_state_{}_{}.pth".format(epoch, step)))

            # save full experiment state at the end of each epoch
            checkpoint = {
                'epoch': epoch + 1,
                'model_curr_state_dict': self.model.state_dict(),
                'model_best_state_dict': self.model_best_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': None if self.scheduler is None else self.scheduler.state_dict(),
                'no_score_improvement': self.no_score_improvement,
                'best_validation_metric': self.best_validation_metric,
            }
            torch.save(checkpoint, join(self.save_path, "full_state.pth"))

        metrics_train = dict([(key + "_train", val) for key, val in metrics_train.items()])
        metrics_valid = dict([(key + "_valid", val) for key, val in metrics_valid.items()])
        time_delta = datetime.now() - self.now
        s = self.format_str.format(time_delta=time_delta,
                                   epoch=epoch,
                                   step=step,
                                   max_epoch=self.max_epoch,
                                   **metrics_train,
                                   **metrics_valid)
        self.logger.log(s)
        self.now = datetime.now()
        return False

    def train(self, epoch):
        steps = np.round(np.linspace(0, len(self.dl_train), self.evaluate_freq + 1)).astype(np.int)
        steps = steps[1:-1]
        step = 1
        self.model.train()
        for i, (xb, yb) in enumerate(self.dl_train):
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(xb)
            if out.ndim == 2 and out.shape[1] == 1:
                out = out.squeeze(dim=1)
            loss = self.criterion(out, yb)
            loss.backward()
            self.optimizer.step()
            if i in steps:
                res = self.evaluate(epoch, step)
                if res:
                    return True
                step += 1
                self.model.train()

        self.evaluate(epoch, step)
        return False

    def run(self):
        self.reset()
        experiment_start = datetime.now()

        if self.save_path is not None:
            if not isdir(self.save_path):
                makedirs(self.save_path)

            # dump all args and their values
            for key, value in self._params.items():
                append_to_file(join(self.save_path, "params.txt"), "{}: {}".format(key, repr(value)))

        if self.load_path is not None:
            # load full experiment state to continue experiment
            load_path = join(self.load_path, "full_state.pth")
            if not isfile(load_path):
                raise ValueError("Checkpoint file {} does not exist".format(load_path))

            checkpoint = torch.load(load_path)

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.model_best_state_dict = checkpoint['model_best_state_dict']
            self.model.load_state_dict(checkpoint['model_curr_state_dict'])

            self.init_epoch = checkpoint['epoch']
            self.best_validation_metric = checkpoint['best_validation_metric']

            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.logger.log("Successfully loaded checkpoint.")

        self.logger.log(self.format_str)

        self.now = datetime.now()
        for epoch in range(self.init_epoch, self.max_epoch):
            res = self.train(epoch)
            if res:
                break

        self.logger.log("Experiment time: {}".format(datetime.now() - experiment_start))
        return self.results

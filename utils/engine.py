import torch
from torchmetrics import metric


class Engine:
    @staticmethod
    def train_batch(model, data, optimizer, criterion, metrics):
        model.train()

        metric_results = dict()

        x, y_true = data
        y_pred = model(x)
        optimizer.zero_grad()

        loss, acc = criterion(y_pred, y_true)
        metric_results["loss"] = loss
        metric_results["acc"] = acc

        for key, metric in metrics.items():
            metric_results[key] = metric(y_pred, y_true)

        loss.backward()
        optimizer.step()

        return metric_results

    @torch.no_grad()
    @staticmethod
    def validate_batch(model, data, criterion, metrics):
        model.eval()

        metric_results = dict()

        x, y_true = data
        y_pred = model(x)

        loss, acc = criterion(y_pred, y_true)
        metric_results["loss"] = loss
        metric_results["acc"] = acc

        for key, metric in metrics.items():
            metric_results[key] = metric(y_pred, y_true)

        return metric_results

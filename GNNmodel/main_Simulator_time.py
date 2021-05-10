# %%
import os
import sys
import glob
import argparse
import time
import logging
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler
import torch_geometric.data as PyGdata
import random

from datasets.NTIGA_simulator_time import NTIGADataset_simulator, NTIGADataset_new
from model.GN_simulator_time import Simulator, SimulatorModel
import utils.metric as metric
import utils.io as io


class StreamToLogger(object):
    """
      Fake file-like stream object that redirects writes to a logger instance.
      """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def SetupLogger(args):
    if args.resume:
        logging.basicConfig(
            level=logging.DEBUG,
            filename=args.log_file,
            filemode="a+",
            datefmt="%Y/%m/%d %H:%M:%S",
            format="%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.DEBUG,
            filename=args.log_file,
            filemode="w+",
            datefmt="%Y/%m/%d %H:%M:%S",
            format="%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s",
        )

    logger = logging.getLogger(__name__)

    stdout_logger = logging.getLogger("STDOUT")
    sl = StreamToLogger(stdout_logger, logging.DEBUG)
    sys.stdout = sl

    console = logging.StreamHandler()
    # optional, set the logging level
    console.setLevel(logging.DEBUG)
    # set a format which is the same for console use
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logger.addHandler(console)
    return logger


def split_ids(ids, folds=5):
    n_samples = len(ids)

    fold_sizes = np.full(folds, n_samples // folds, dtype=np.int)
    fold_sizes[: n_samples % folds] += 1
    current = 0

    train_ids = []
    test_ids = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_ids.append(ids[start:stop])
        train_id = ids[:start] + ids[stop:]
        train_ids.append(train_id)
        current = stop
    return train_ids, test_ids


def train_epoch(
    epoch,
    device,
    train_data_loader,
    model,
    flg_simulator,
    criterion,
    optimizer,
    scheduler,
    logger,
    log_interval,
    checkpoint_path,
):
    epoch_loss = 0.0
    avg_error = 0.0
    avg_RAE = 0.0
    t0 = time.time()
    model.train()

    for iteration, batch in enumerate(train_data_loader, 1):
        batch = batch.to(device)
        optimizer.zero_grad()

        list_y = []
        edge = batch.edge_index
        # y = batch.x[:, -1, 0]
        y = batch.y[:, :, 0].squeeze()
        # print(batch.x[:, -1, 0].shape)
        # print(y.shape.)
        num_tstep = batch.x.shape[-1] - 1

        random.seed(1)  # Teaching force

        for k in range(num_tstep):

            rand_val = random.random()
            X_curr = batch.x[:, :, k]
            if rand_val > 0 or k == 0:
                y = model(X_curr, edge, y, 1)
            else:
                y = model(X_curr, edge, batch.y[:, :, k - 1].squeeze(), 1)

            list_y.append(y)

        prediction = torch.stack(list_y, dim=1)
        target = batch.y[:, :, 1 : (num_tstep + 1)].squeeze()

        loss = criterion(prediction, target)

        epoch_loss += loss.item() * batch.num_graphs
        loss.backward()
        optimizer.step()

        prediction = (
            prediction.view(batch.num_graphs, -1, num_tstep)
            .permute(0, 2, 1)
            .reshape(batch.num_graphs * num_tstep, -1)
        )
        target = (
            target.view(batch.num_graphs, -1, num_tstep)
            .permute(0, 2, 1)
            .reshape(batch.num_graphs * num_tstep, -1)
        ).detach()
        tmp_error = metric.ComputeTestErrorMAE(prediction, target)
        avg_error += torch.sum(tmp_error) / num_tstep

        # print("Epoch: {:d} Iteration: {:d} Finish!".format(epoch, iteration))

    if (epoch + 1) % log_interval == 0:
        if flg_simulator == 1:
            fname_checkpoint = checkpoint_path + "pipe_epoch_{}.pth".format(epoch)
        elif flg_simulator == 2:
            fname_checkpoint = checkpoint_path + "bifur_epoch_{}.pth".format(epoch)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss,
            },
            fname_checkpoint,
        )
    dur = time.time() - t0

    return dur, epoch_loss, avg_error


def test_epoch(
    device, test_data_loader, model, flg_simulator, criterion,
):
    epoch_loss = 0.0
    avg_error = 0.0
    avg_RAE = 0.0
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        avg_error = 0
        for iteration, batch in enumerate(test_data_loader, 1):

            batch = batch.to(device)

            list_y = []
            edge = batch.edge_index
            # y = batch.x[:, -1, 0]
            # num_tstep = batch.x.shape[-1]

            y = batch.y[:, :, 0].squeeze()
            num_tstep = batch.x.shape[-1] - 1

            for k in range(num_tstep):

                X_curr = batch.x[:, :, k]
                y = model(X_curr, edge, y, 1)

                list_y.append(y)

            prediction = torch.stack(list_y, dim=1)

            # target = batch.y.squeeze()
            target = batch.y[:, :, 1 : (num_tstep + 1)].squeeze()

            loss = criterion(prediction, target)

            epoch_loss += loss.item() * batch.num_graphs

            prediction = (
                prediction.view(batch.num_graphs, -1, num_tstep)
                .permute(0, 2, 1)
                .reshape(batch.num_graphs * num_tstep, -1)
            )
            target = (
                target.view(batch.num_graphs, -1, num_tstep)
                .permute(0, 2, 1)
                .reshape(batch.num_graphs * num_tstep, -1)
            ).detach()
            tmp_error = metric.ComputeTestErrorMAE(prediction, target)
            avg_error += torch.sum(tmp_error) / num_tstep

            batch = batch.to(device)

    dur = time.time() - t0
    return dur, epoch_loss, avg_error


def train(args):

    flg_simulator = args.flg_sim  # 1: Pipe; 2: Bifurcation
    log_interval = args.log_interval

    # * Setting up work directory and logging
    checkpoint_path = args.checkpt_path
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    logger = SetupLogger(args)

    # * Setting up training device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    # print(device)
    logger.info(device)

    # * Load dataset
    if args.data_path is not None:
        data_path = args.data_path
        dataset = NTIGADataset_new(
            file_path=data_path,
            recursive=False,
            load_data=True,
            data_cache_size=2000000,
        )
        # print("Dataset: ", data_path)
        logger.info("Dataset: {}".format(data_path))
    if args.data_fname is not None:
        data_fname = args.data_fname
        dataset = NTIGADataset_simulator(data_fname)
        # print("Dataset: ", data_fname)
        logger.info("Dataset: {}".format(data_fname))

    # * Setting up dataloader
    torch.manual_seed(args.seed)
    train_loader_params = (
        {
            "batch_size": args.batch_size,
            "num_workers": args.num_worker,
            "pin_memory": True,
        }
        if args.cuda
        else {}
    )
    test_loader_params = (
        {
            "batch_size": args.test_batch_size,
            "num_workers": args.num_worker,
            "pin_memory": True,
        }
        if args.cuda
        else {}
    )

    ids = list(range(len(dataset)))
    if args.shuffle_dataset:
        np.random.seed(args.seed)
        np.random.shuffle(ids)

    train_ids, test_ids = split_ids(ids, args.folds)
    logger.info("Num of samples: {:d}".format(len(dataset)))

    # * Setting up training parameters
    num_epochs = args.epochs

    model = SimulatorModel(
        num_features=dataset.num_in,
        num_targets=dataset.num_out,
        num_layers=args.num_layers,
        num_hidden=args.num_hidden,
    )
    model.to(device)
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_stepsize, gamma=args.lr_gama
    )
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gama)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     "min",
    #     factor=args.lr_gama,
    #     patience=0,
    #     threshold=1e-4,
    #     min_lr=1e-6,
    #     verbose=True,
    # )
    last_epoch = -1
    # * Read parameters from last checkpoint before continuing training
    if args.resume:
        list_of_files = glob.glob(checkpoint_path + "*.pth")
        latest_file = max(list_of_files, key=os.path.getctime)

        # print("Checkpoint file: ", latest_file)
        logger.info("Checkpoint file: {}".format(latest_file))
        load_checkpoint = latest_file
        checkpoint = torch.load(load_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        last_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

    # * Start training (single fold)
    if not args.kfold_cv:
        dur = []
        train_id = train_ids[args.id_fold]
        test_id = test_ids[args.id_fold]
        train_sampler = SubsetRandomSampler(train_id)
        test_sampler = SubsetRandomSampler(test_id)
        train_data_loader = PyGdata.DataLoader(
            dataset, sampler=train_sampler, **train_loader_params
        )
        test_data_loader = PyGdata.DataLoader(
            dataset, sampler=test_sampler, **test_loader_params
        )
        logger.info("Num of train samples: {:d}".format(len(train_id)))
        logger.info("Num of batches: {:d}".format(len(train_data_loader)))
        for epoch in range(last_epoch + 1, last_epoch + 1 + num_epochs):
            logger.debug(
                "Epoch: {:d} Start! LR: {:.7f}".format(epoch, scheduler.get_lr()[0])
            )
            epoch_train_dur, epoch_train_loss, avg_train_error = train_epoch(
                epoch,
                device,
                train_data_loader,
                model,
                flg_simulator,
                criterion,
                optimizer,
                scheduler,
                logger,
                log_interval,
                checkpoint_path,
            )
            epoch_test_dur, epoch_test_loss, avg_test_error = test_epoch(
                device, test_data_loader, model, flg_simulator, criterion
            )
            logger.debug(
                "Epoch: {:d} Time(s):{:.4f} Train Loss: {:.4f} Train MAE:{:.4f}".format(
                    epoch,
                    epoch_train_dur,
                    epoch_train_loss / len(train_id),
                    avg_train_error / len(train_id),
                )
            )
            logger.debug(
                "Epoch: {:d} Time(s):{:.4f} Test Loss: {:.4f} Test MAE:{:.4f}".format(
                    epoch,
                    epoch_test_dur,
                    epoch_test_loss / len(test_id),
                    avg_test_error / len(test_id),
                )
            )

            scheduler.step()
            # scheduler.step(avg_test_error)  #! step for ReduceLROnPlateau

    # * Start training (sequential k-fold cross validation)
    else:
        for k in range(args.folds):
            train_id = train_ids[k]
            test_id = test_ids[k]
            train_sampler = SubsetRandomSampler(train_id)
            test_sampler = SubsetRandomSampler(test_id)
            train_data_loader = PyGdata.DataLoader(
                dataset, sampler=train_sampler, **train_loader_params
            )
            test_data_loader = PyGdata.DataLoader(
                dataset, sampler=test_sampler, **test_loader_params
            )
            logger.info("Num of batches: {:d}".format(len(train_data_loader)))

            for epoch in range(last_epoch + 1, last_epoch + 1 + num_epochs):
                # logger.debug(
                #     "Epoch: {:d} Start! LR: {:.7f}".format(epoch, scheduler.get_lr()[0])
                # )
                epoch_train_dur, epoch_train_loss, avg_train_error = train_epoch(
                    epoch,
                    device,
                    train_data_loader,
                    model,
                    flg_simulator,
                    criterion,
                    optimizer,
                    scheduler,
                    logger,
                    log_interval,
                    checkpoint_path,
                )
                epoch_test_dur, epoch_test_loss, avg_test_error = test_epoch(
                    device, test_data_loader, model, flg_simulator, criterion
                )
                logger.debug(
                    "Epoch: {:d} Time(s):{:.4f} Train Loss: {:.4f} Train MAE:{:.4f}".format(
                        epoch,
                        epoch_train_dur,
                        epoch_train_loss / len(train_id),
                        avg_train_error / len(train_id),
                    )
                )
                logger.debug(
                    "Epoch: {:d} Time(s):{:.4f} Test Loss: {:.4f} Test MAE:{:.4f}".format(
                        epoch,
                        epoch_test_dur,
                        epoch_test_loss / len(test_id),
                        avg_test_error / len(test_id),
                    )
                )
                scheduler.step()


def evaluation(args):
    # device_load = torch.device("cpu")
    # checkpoint = torch.load(fname_checkpoint, map_location=device_load)

    flg_simulator = args.flg_sim  # 1: Pipe; 2: Bifurcation
    log_interval = args.log_interval

    # * Setting up evaluation device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    # * Setting up dataloader
    if args.data_path is not None:
        data_path = args.data_path
        dataset = NTIGADataset_new(
            file_path=data_path,
            recursive=False,
            load_data=True,
            data_cache_size=2000000,
        )
        print("Dataset: ", data_path)

    if args.data_fname is not None:
        data_fname = args.data_fname
        dataset = NTIGADataset_simulator(data_fname)
        print("Dataset: ", data_fname)

    torch.manual_seed(args.seed)
    loader_params = (
        {
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": True,
        }
        if args.cuda
        else {}
    )

    data_loader = PyGdata.DataLoader(dataset, **loader_params)

    print("Num of samples: ", len(dataset))
    print("Num of batches: ", len(data_loader))

    # * Setting up model for evaluation
    model = SimulatorModel(
        num_features=dataset.num_in,
        num_targets=dataset.num_out,
        num_layers=args.num_layers,
        num_hidden=args.num_hidden,
    ).to(device)
    criterion = torch.nn.MSELoss()

    checkpoint_fname = args.checkpt_fname
    checkpoint = torch.load(checkpoint_fname)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    model.eval()
    print("Evalutiona checkpoint: ", checkpoint_fname)

    t0 = time.time()
    with torch.no_grad():
        epoch_loss = 0
        avg_error = 0
        for iteration, batch in enumerate(data_loader, 1):
            batch = batch.to(device)

            # prediction, _, _ = model(batch, flg_simulator)  # ! Graph Net
            prediction = model(batch, flg_simulator)
            prediction = prediction.detach()
            prediction = prediction.view(batch.num_graphs, -1)

            target = batch.y
            target = target.detach()
            target = target.view(batch.num_graphs, -1)

            tmp_error = metric.ComputeTestErrorRMSE(prediction, target)

            avg_error += torch.sum(tmp_error)

            # loss = criterion(model(batch, flg_simulator)[0], batch.y)  # ! Graph Net
            loss = criterion(model(batch, flg_simulator), batch.y)

            epoch_loss += loss.item() * batch.num_graphs
        epoch_loss /= len(dataset)
        avg_error /= len(dataset)

        print(
            "Epoch: {:d} Time(s):{:.4f} Loss: {:.4f} Error:{:.4f} ".format(
                epoch, time.time() - t0, epoch_loss, avg_error,
            )
        )


def prediction(args):
    flg_simulator = args.flg_sim  # 1: Pipe; 2: Bifurcation
    log_interval = args.log_interval

    predict_path = args.predict_path
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    # * Setting up evaluation device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    logger = SetupLogger(args)

    # * Load dataset
    if args.data_path is not None:
        data_path = args.data_path
        dataset = NTIGADataset_new(
            file_path=data_path,
            recursive=False,
            load_data=True,
            data_cache_size=2000000,
        )
        # print("Dataset: ", data_path)
        logger.info("Dataset: {}".format(data_path))
    if args.data_fname is not None:
        data_fname = args.data_fname
        dataset = NTIGADataset_simulator(data_fname)
        # print("Dataset: ", data_fname)
        logger.info("Dataset: {}".format(data_fname))

    # * Setting up dataloader
    torch.manual_seed(args.seed)
    train_loader_params = (
        {
            "batch_size": args.batch_size,
            "num_workers": args.num_worker,
            "pin_memory": True,
        }
        if args.cuda
        else {}
    )
    test_loader_params = (
        {
            "batch_size": args.test_batch_size,
            "num_workers": args.num_worker,
            "pin_memory": True,
        }
        if args.cuda
        else {}
    )

    ids = list(range(len(dataset)))
    if args.shuffle_dataset:
        np.random.seed(args.seed)
        np.random.shuffle(ids)

    train_ids, test_ids = split_ids(ids, args.folds)

    train_id = train_ids[args.id_fold]
    test_id = test_ids[args.id_fold]

    # * Setting up dataloader
    loader_params = (
        {
            "batch_size": args.batch_size,
            "num_workers": args.num_worker,
            "pin_memory": True,
        }
        if args.cuda
        else {}
    )

    logger.info("Num of samples: {:d}".format(len(dataset)))

    data_loader = PyGdata.DataLoader(dataset, **loader_params)

    print("Num of samples: ", len(dataset))
    # print("Num of batches: ", len(data_loader))

    # * Setting up model for evaluation
    model = SimulatorModel(
        num_features=dataset.num_in,
        num_targets=dataset.num_out,
        num_layers=args.num_layers,
        num_hidden=args.num_hidden,
    ).to(device)
    criterion = torch.nn.MSELoss()

    checkpoint_fname = args.checkpt_fname
    checkpoint = torch.load(checkpoint_fname)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    model.eval()
    print("Evalutiona checkpoint: ", checkpoint_fname)

    t0 = time.time()
    with torch.no_grad():
        epoch_loss = 0
        avg_error = 0

        # # for idx in range(len(test_id)):
        # for idx in range(len(test_id)):
        #     print(dataset[test_id[idx]].y[0, -1, 0])

        for idx in range(6, 9):
            # for idx in range(len(test_id)):
            data_sample = dataset[test_id[idx]].to(device)
            # data_sample = dataset[train_id[idx]].to(device)
            pts = data_sample.x[:, :3, 0]
            num_tstep = data_sample.x.shape[-1] - 1
            edge = data_sample.edge_index
            # y = data_sample.x[:, -1, 0]
            y = data_sample.y[:, :, 0].squeeze()

            error = 0
            # for k in range(num_tstep/2):
            for k in range(50):
                X_curr = data_sample.x[:, :, k]
                y = model(X_curr, edge, y, 1)

                # print(data_sample.y[:, :, k].squeeze().shape)

                # print(criterion(y, data_sample.y[:, :, k].squeeze()))
                # print(metric.ComputeTestErrorMAE(y, data_sample.y[:, :, k].squeeze()))
                # print(y.unsqueeze(-1).size())
                # print(data_sample.y[:, :, k].size())

                # tmp_error = torch.mean(
                #     metric.ComputeTestErrorMAE(y.unsqueeze(-1), data_sample.y[:, :, k])
                # )
                tmp_error = torch.mean(
                    metric.ComputeTestErrorMAE(
                        y.unsqueeze(-1), data_sample.y[:, :, k + 1]
                    )
                )
                error += tmp_error

                fname_output_vtk = predict_path + "sample{}_output_{}.vtk".format(
                    idx, k
                )
                # io.WriteVTK(fname_output_vtk, pts, output)
                io.WriteVTK(fname_output_vtk, pts, y)

                fname_target_vtk = predict_path + "sample{}_target_{}.vtk".format(
                    idx, k
                )
                # io.WriteVTK(fname_target_vtk, pts, data_sample.y)
                io.WriteVTK(fname_target_vtk, pts, data_sample.y[:, :, k + 1].squeeze())
            print("Idx:{} BC value {}".format(idx, dataset[test_id[idx]].y[0, -1, 0]))
            print("Idx:{} Avg Error {}".format(idx, error / num_tstep))
        # for i in range(0, 40):
        #     data_sample = dataset[50099 + i * 10].to(device)
        #     pts = data_sample.x[:, :3]

        #     # output = model(data_sample, flg_simulator)[0]  # ! Graph Net
        #     output = model(data_sample, flg_simulator)
        #     print(criterion(output, data_sample.y))
        #     print(metric.ComputeTestErrorMAE(output, data_sample.y))
        #     # fname_output = "/export/home/math/angranl/Documents/NeuronTransportLearning/data/Pipe/feature_map/output{}.txt".format(i)
        #     fname_output_vtk = predict_path + "output{}.vtk".format(i)
        #     # io.WriteVTK(fname_output_vtk, pts, output)
        #     io.WriteVTK(fname_output_vtk, pts, output)

        #     fname_target_vtk = predict_path + "target{}.vtk".format(i)
        #     # io.WriteVTK(fname_target_vtk, pts, data_sample.y)
        #     io.WriteVTK(fname_output_vtk, pts, output)

        # for i in range(0, 40):
        #     data_sample = dataset[50099 + i * 10].to(device)
        #     pts = data_sample.x[:, :3]

        #     # output = model(data_sample, flg_simulator)[0]  # ! Graph Net
        #     output = model(data_sample, flg_simulator)
        #     print(criterion(output, data_sample.y))
        #     print(metric.ComputeTestErrorMAE(output, data_sample.y))
        #     # fname_output = "/export/home/math/angranl/Documents/NeuronTransportLearning/data/Pipe/feature_map/output{}.txt".format(i)
        #     fname_output_vtk = predict_path + "output{}.vtk".format(i)
        #     # io.WriteVTK(fname_output_vtk, pts, output)
        #     io.WriteVTK(fname_output_vtk, pts, output)

        #     fname_target_vtk = predict_path + "target{}.vtk".format(i)
        #     # io.WriteVTK(fname_target_vtk, pts, data_sample.y)
        #     io.WriteVTK(fname_output_vtk, pts, output)

        # with open(fname_target, "w") as outF:
        #         for x in data_sample.y.to("cpu").tolist():
        #             print(*x, file=outF, sep = ' ')

        # for iteration, batch in enumerate(data_loader, 1):
        #     batch = batch.to(device)

        #     prediction = model(batch, flg_simulator)
        #     prediction = prediction.detach()
        #     prediction = prediction.view(batch.num_graphs, -1)

        #     target = batch.y
        #     target = target.detach()
        #     target = target.view(batch.num_graphs, -1)

        #     tmp_error = metric.ComputeTestErrorRMSE(prediction, target)

        #     avg_error += torch.sum(tmp_error)

        #     loss = criterion(model(batch, flg_simulator), batch.y)

        #     epoch_loss += loss.item() * batch.num_graphs
        # epoch_loss /= len(dataset)
        # avg_error /= len(dataset)

        # print(
        #     "Epoch: {:d} Time(s):{:.4f} Loss: {:.4f} Error:{:.4f} ".format(
        #         epoch, time.time() - t0, epoch_loss, avg_error,
        #     )
        # )


def main(args):
    if args.train:
        train(args)

    if args.eval:
        evaluation(args)

    if args.predict:
        prediction(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Neuron Simulator")
    parser.add_argument(
        "--train", action="store_true", default=False, help="enables training"
    )
    parser.add_argument(
        "--eval", action="store_true", default=False, help="enables evaluation"
    )
    parser.add_argument(
        "--predict", action="store_true", default=False, help="enables prediction"
    )
    parser.add_argument(
        "--resume", action="store_true", default=False, help="resume training"
    )
    parser.add_argument(
        "--kfold-cv", action="store_true", default=False, help="enable cross validation"
    )
    parser.add_argument(
        "--shuffle-dataset",
        action="store_true",
        default=False,
        help="enable dataset shuffle",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        metavar="path",
        help="path to the dataset",
    )
    parser.add_argument(
        "--data-fname",
        type=str,
        default=None,
        metavar="fname",
        help="path to the dataset file",
    )
    parser.add_argument(
        "--checkpt-path",
        type=str,
        default=None,
        metavar="path",
        help="path to save checkpoint files",
    )
    parser.add_argument(
        "--checkpt-fname",
        type=str,
        default=None,
        metavar="path",
        help="path to the checkpt for evaluation",
    )
    parser.add_argument(
        "--predict-path",
        type=str,
        default=None,
        metavar="path",
        help="path to save prediction files",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--flg-sim",
        type=int,
        default=1,
        metavar="N",
        help="simulator to be trained (default: 1) ",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="enables CUDA training"
    )

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        metavar="N",
        help="number of cross-validation folds",
    )
    parser.add_argument(
        "--id-fold",
        type=int,
        default=0,
        metavar="N",
        help="index of the fold for training and testing",
    )
    parser.add_argument(
        "--num-worker",
        type=int,
        default=0,
        metavar="N",
        help="num_worker for Dataloader",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        metavar="N",
        help="number of hidden layers",
    )
    parser.add_argument(
        "--num-hidden",
        type=int,
        default=32,
        metavar="N",
        help="the size of hidden layer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=512,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=10,
        metavar="fname",
        help="set up file to record training process",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="N",
        help="set (initial) learning rate",
    )
    parser.add_argument(
        "--lr-gama", type=float, default=0.9, metavar="N", help="gama for lr scheduler",
    )
    parser.add_argument(
        "--lr-stepsize",
        type=int,
        default=2,
        metavar="N",
        help="step size for lr scheduler",
    )
    args = parser.parse_args()

    print(args)

    main(args)


# if flg_prediction:


# %%

# # ! Create feature map of each layer
# if flg_postprocess:
#     fname_checkpoint = checkpoint_path + "epoch99.pth"
#     checkpoint = torch.load(fname_checkpoint)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     epoch = checkpoint["epoch"]
#     loss = checkpoint["loss"]

#     activation = {}

#     def get_activation(name):
#         def hook(model, input, output):
#             activation[name] = output.detach()

#         return hook

#     # model.conv1.register_forward_hook(get_activation("conv1"))
#     # model.conv2.register_forward_hook(get_activation("conv2"))
#     # model.conv3.register_forward_hook(get_activation("conv3"))
#     # model.conv4.register_forward_hook(get_activation("conv4"))

#     # data_sample = dataset[49].to(device)
#     # output = model(data_sample)
#     # # print(criterion(output, data_sample.y))

#     # # print(model)

#     # # print(activation["conv1"])
#     # # print(activation["conv2"])

#     # for i in range(1, 5):
#     #     fname_feature = "/export/home/math/angranl/Documents/NeuronTransportLearning/data/Pipe/feature_map/conv{}.txt".format(
#     #         i
#     #     )
#     #     hook_name = "conv{}".format(i)
#     #     # model.conv1.register_forward_hook(get_activation(hook_name))
#     #     # output = model(data_sample)
#     #     with open(fname_feature, "w") as outF:
#     #         for x in activation[hook_name].to("cpu").tolist():
#     #             print(*x, file=outF, sep=" ")

#     for i in range(0, 20):
#         data_sample = dataset[49 + i * 100].to(device)
#         pts = data_sample.x[:, :3]
#         output = model(data_sample)
#         print(criterion(output, data_sample.y))
#         print(myutils.ComputeTestError(output, data_sample.y))
#         # fname_output = "/export/home/math/angranl/Documents/NeuronTransportLearning/data/Pipe/feature_map/output{}.txt".format(i)
#         fname_output_vtk = "/export/home/math/angranl/Documents/NeuronTransportLearning/data/Pipe/feature_map/output{}.vtk".format(
#             i
#         )
#         myutils.WriteVTK(fname_output_vtk, pts, output)
#         # with open(fname_output, "w") as outF:
#         #         for x in output.to("cpu").tolist():
#         #             print(*x, file=outF, sep = ' ')

#         # fname_target = "/export/home/math/angranl/Documents/NeuronTransportLearning/data/Pipe/feature_map/target{}.txt".format(i)
#         fname_target_vtk = "/export/home/math/angranl/Documents/NeuronTransportLearning/data/Pipe/feature_map/target{}.vtk".format(
#             i
#         )
#         myutils.WriteVTK(fname_target_vtk, pts, data_sample.y)
#         # with open(fname_target, "w") as outF:
#         #         for x in data_sample.y.to("cpu").tolist():
#         #             print(*x, file=outF, sep = ' ')
# %%
# model.encoder[1].register_forward_hook(get_activation('ext_conv1'))
# output = model(input)
# act = activation['ext_conv1'].squeeze()

# model.eval()
# loss = criterion(out, data.y)

# print('Accuracy: {:.4f}'.format(loss))

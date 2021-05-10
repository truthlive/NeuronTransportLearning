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

from datasets.NTIGA_simulator import NTIGADataset_simulator, NTIGADataset_new
from datasets.NTIGA_tree import NTIGADataset_tree
from model.GN_simulator import Simulator, SimulatorModel
from model.GN_tree import GN_tree
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

        prediction = model(batch, flg_simulator)
        prediction = prediction.detach()
        prediction = prediction.view(batch.num_graphs, -1)

        target = batch.y
        target = target.detach()
        target = target.view(batch.num_graphs, -1)

        # print(prediction.shape)
        tmp_error = metric.ComputeTestErrorMAE(prediction, target)
        avg_error += torch.sum(tmp_error)

        # tmp_RAE = myutils.ComputeTestErrorRAE(prediction, target)
        # avg_RAE += torch.sum(tmp_RAE)

        loss = criterion(model(batch, flg_simulator), batch.y)

        epoch_loss += loss.item() * batch.num_graphs
        loss.backward()
        optimizer.step()
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

            prediction = model(batch, flg_simulator)
            prediction = prediction.detach()
            prediction = prediction.view(batch.num_graphs, -1)

            target = batch.y
            target = target.detach()
            target = target.view(batch.num_graphs, -1)

            tmp_error = metric.ComputeTestErrorMAE(prediction, target)

            avg_error += torch.sum(tmp_error)

            loss = criterion(model(batch, flg_simulator), batch.y)

            epoch_loss += loss.item() * batch.num_graphs

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
        dataset = NTIGADataset_tree(data_fname)
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

    # * Load pretrained simulator
    fname_sim_pre = args.sim_pre_fname
    sim_pre = torch.load(fname_sim_pre, map_location=torch.device("cpu"))

    gn_sim = SimulatorModel(5, 1, 10, 32)
    gn_sim.load_state_dict(sim_pre["model_state_dict"])

    # * Setting up training parameters
    num_epochs = args.epochs

    model = GN_tree(gn_sim, num_layers=args.num_layers, num_hidden=args.num_hidden)
    model.to(device)
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_stepsize, gamma=args.lr_gama
    )
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
    model = Simulator(
        dataset=dataset, num_layers=args.num_layers, num_hidden=args.num_hidden
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
    print("Num of batches: ", len(data_loader))

    # * Load pretrained simulator
    fname_sim_pre = args.sim_pre_fname
    sim_pre = torch.load(fname_sim_pre, map_location=torch.device("cpu"))

    gn_sim = SimulatorModel(5, 1, 10, 32)
    gn_sim.load_state_dict(sim_pre["model_state_dict"])


    # * Setting up model for evaluation

    model = GN_tree(gn_sim, num_layers=args.num_layers, num_hidden=args.num_hidden)
    model.to(device)
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

        for i in range(0, 40):
            data_sample = dataset[50099 + i * 10].to(device)
            pts = data_sample.x[:, :3]

            # output = model(data_sample, flg_simulator)[0]  # ! Graph Net
            output = model(data_sample, flg_simulator)
            print(criterion(output, data_sample.y))
            print(metric.ComputeTestErrorMAE(output, data_sample.y))
            # fname_output = "/export/home/math/angranl/Documents/NeuronTransportLearning/data/Pipe/feature_map/output{}.txt".format(i)
            fname_output_vtk = predict_path + "output{}.vtk".format(i)
            # io.WriteVTK(fname_output_vtk, pts, output)
            io.WriteVTK(fname_output_vtk, pts, output)

            # fname_bcini_vtk = predict_path + "bc_initial{}.vtk".format(i)
            # print(type(data_sample.y))
            # print(type(data_sample.x[:, -1]))
            # bc_ini = data_sample.x[:, -1]
            # io.WriteVTK_bcini(fname_bcini_vtk, pts, bc_ini)
            # with open(fname_output, "w") as outF:
            #         for x in output.to("cpu").tolist():
            #             print(*x, file=outF, sep = ' ')

            # fname_target = "/export/home/math/angranl/Documents/NeuronTransportLearning/data/Pipe/feature_map/target{}.txt".format(i)
            fname_target_vtk = predict_path + "target{}.vtk".format(i)
            # io.WriteVTK(fname_target_vtk, pts, data_sample.y)
            io.WriteVTK(fname_output_vtk, pts, output)

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
        "--sim-pre-fname",
        type=str,
        default=None,
        metavar="path",
        help="path to the pretrained simulator model",
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

# argument parser
import argparse

# Misc.
import os
import pickle
import random
from typing import Tuple

# JAX
import jax
import jax.numpy as jnp
import numpy as np

# PyTorch
import torch
import torch.nn.functional as F
from jax import grad, jit, lax, vmap
from jax.example_libraries import optimizers, stax
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# Disable GPU Preallocation for safe multiprocessing
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# JAX Network Definition
def make_network(mode="train"):
    # Use stax to set up the model
    # Same as pytorch_mnist.py
    model_init, model_apply = stax.serial(
        stax.Conv(out_chan=32, filter_shape=(3, 3), strides=(1, 1), padding="VALID"),
        stax.Relu,
        stax.Conv(out_chan=64, filter_shape=(3, 3), strides=(1, 1), padding="VALID"),
        stax.Relu,
        stax.MaxPool(window_shape=(2, 2)),
        # stax.Dropout(rate=0.25, mode=mode),
        stax.Flatten,
        stax.Dense(out_dim=128),
        stax.Relu,
        # stax.Dropout(rate=0.5, mode=mode),
        stax.Dense(out_dim=10),
        stax.LogSoftmax,
    )
    return model_init, model_apply


def train(args, get_params, loss_fn, opt_state, update_fn, train_loader, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 1. cast to jax.array
        data = jnp.asarray(np.asarray(data)).transpose(0, 2, 3, 1)
        target = jnp.asarray(np.asarray(target))
        # 2. get parameter
        params = get_params(opt_state)
        # 3. perform prediction
        loss, gradient = jax.value_and_grad(loss_fn)(params, data, target)
        # 4. update weights through backpropagation
        opt_state = update_fn(batch_idx, gradient, opt_state)

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break
    return opt_state


def test(params, acc_fn, loss_fn, test_loader):
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = jnp.asarray(np.asarray(data)).transpose(0, 2, 3, 1)
        target = jnp.asarray(np.asarray(target))
        loss, gradient = jax.value_and_grad(loss_fn)(params, data, target)
        acc = acc_fn(params, data, target)
        test_loss += loss * len(data)
        correct += acc
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="JAX MNIST Example")
    parser.add_argument("--key", type=int, default=0, metavar="N", help="main key for JAX PRNG")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="quickly check a single pass"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        metavar="N",
        help="how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.(default: 6)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="For Saving the current Model"
    )
    args = parser.parse_args()

    # Fix randomness
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    key = jax.random.PRNGKey(args.key)

    train_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": False,
        "shuffle": True,
    }
    test_kwargs = {
        "batch_size": args.test_batch_size,
        "num_workers": args.num_workers,
        "pin_memory": False,
        "shuffle": False,
    }

    # See https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/20
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    target_transform = transforms.Compose(
        [lambda x: torch.tensor(x), lambda x: F.one_hot(x, num_classes=10)]
    )
    dataset1 = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )
    dataset2 = datasets.MNIST(
        "../data", train=False, transform=transform, target_transform=target_transform
    )
    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    # Create Network
    model_init, model_apply = make_network(mode="train")

    # Forward - Logits
    model_logit = jit(lambda params, x: model_apply(params, x))
    # Prediction
    model_pred = jit(lambda params, x: jnp.argmax(model_logit(params, x), axis=1))
    # Accuracy
    model_acc = jit(lambda params, x, y: jnp.sum(model_pred(params, x) == jnp.argmax(y, axis=1)))
    # NLL(Negative Log Likelihood) Loss
    model_loss = jit(lambda params, x, y: -jnp.mean(jnp.sum(model_logit(params, x) * y, axis=1)))

    # We have defined our model. We need to initialze the params based on the input shape.
    # The images in our dataset are of shape (28, 28, 1). Hence we will initialize the
    # network with the input shape (-1, 28, 28, 1). -1 represents the batch dimension here
    model_output_shape, model_params = model_init(key, input_shape=(-1, 28, 28, 1))
    # Create Optimizer
    opt_init, opt_update, get_params = optimizers.adam(args.lr)
    opt_update = jit(opt_update)
    opt_state = opt_init(model_params)

    for epoch in range(1, args.epochs + 1):
        opt_state = train(args, get_params, model_loss, opt_state, opt_update, train_loader, epoch)
        test(get_params(opt_state), model_acc, model_loss, test_loader)

    if args.save_model:
        # Ref : https://stackoverflow.com/questions/64550792/how-do-i-save-an-optimizer-state-of-jax-trained-model
        with open("mnist_cnn.pkl", "wb") as f:
            pickle.dump(optimizers.unpack_optimizer_state(opt_state), f)


def load_checkpoint(ckpt_path):
    with open("mnist_cnn.pkl", "rb") as f:
        params = pickle.load(f)
    opt_state = optimizers.pack_optimizer_state(params)
    return opt_state


if __name__ == "__main__":
    main()

import gzip
import os
import shutil
import struct
import sys
import time

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import datasets, transforms

# Add the build directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "build", "python"))
import data_loader


def verify_mnist_file(file_path, expected_magic_number):
    """Verify if a MNIST file is valid"""
    try:
        with open(file_path, "rb") as f:
            magic = struct.unpack(">I", f.read(4))[0]
            if magic != expected_magic_number:
                print(
                    f"Warning: {file_path} has incorrect magic number: {magic} (expected {expected_magic_number})"
                )
                return False
            return True
    except Exception as e:
        print(f"Error verifying {file_path}: {str(e)}")
        return False


def download_file(url, file_path):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        progress_bar_width = 50

        with open(file_path, "wb") as f:
            downloaded = 0
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded += len(data)
                if total_size > 0:
                    progress = int(progress_bar_width * downloaded / total_size)
                    print(
                        f"\rDownloading: [{'=' * progress}{' ' * (progress_bar_width - progress)}] {downloaded}/{total_size} bytes",
                        end="",
                    )
            print()  # New line after progress bar
    except Exception as e:
        print(f"\nError downloading file: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise


def download_mnist_files():
    """Download MNIST files if they don't exist"""
    # Updated MNIST URLs
    base_urls = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    files = {
        "train-images-idx3-ubyte.gz": (
            "train-images-idx3-ubyte",
            2051,
        ),  # Magic number for images
        "train-labels-idx1-ubyte.gz": (
            "train-labels-idx1-ubyte",
            2049,
        ),  # Magic number for labels
    }

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "mnist")
    os.makedirs(data_dir, exist_ok=True)

    for gz_file, (target_file, magic_number) in files.items():
        target_path = os.path.join(data_dir, target_file)

        # Check if file exists and is valid
        if os.path.exists(target_path):
            if not verify_mnist_file(target_path, magic_number):
                print(f"Invalid {target_file}, redownloading...")
                os.remove(target_path)
            else:
                print(f"{target_file} exists and is valid")
                continue

        gz_path = os.path.join(data_dir, gz_file)
        success = False

        # Try different URLs
        for base_url in base_urls:
            try:
                print(f"\nTrying to download from {base_url}")
                download_file(
                    base_url + gz_file, gz_path
                )  # Use the download_file function instead of urllib

                # Extract the gzipped file
                print(f"Extracting {gz_file}...")
                with gzip.open(gz_path, "rb") as f_in:
                    with open(target_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Verify the extracted file
                if verify_mnist_file(target_path, magic_number):
                    print(f"Successfully downloaded and verified {target_file}")
                    success = True
                    break
                else:
                    print(f"Downloaded {target_file} is invalid, trying next URL...")
                    if os.path.exists(gz_path):
                        os.remove(gz_path)
                    if os.path.exists(target_path):
                        os.remove(target_path)

            except Exception as e:
                print(f"Error with URL {base_url}: {str(e)}")
                if os.path.exists(gz_path):
                    os.remove(gz_path)
                if os.path.exists(target_path):
                    os.remove(target_path)
                continue

        if not success:
            raise Exception(
                f"Failed to download valid {target_file} from all available sources"
            )

        # Clean up
        if os.path.exists(gz_path):
            os.remove(gz_path)


def test_custom_data_loader():
    try:
        # Download MNIST files if needed
        download_mnist_files()

        # Get the absolute path to the data directory
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "mnist")
        )

        # Verify files exist
        image_file = os.path.join(data_dir, "train-images-idx3-ubyte")
        label_file = os.path.join(data_dir, "train-labels-idx1-ubyte")

        if not os.path.exists(image_file) or not os.path.exists(label_file):
            raise FileNotFoundError("MNIST files not found after download attempt")

        # Initialize custom data loader
        custom_loader = data_loader.DataLoader(
            data_path=data_dir, batch_size=64, num_workers=4
        )

        # Load MNIST data
        custom_loader.load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte")

        # Test a few batches
        print("Testing custom data loader...")
        start_time = time.time()
        for i in range(10):
            images, labels = custom_loader.get_next_batch()
            print(f"Batch {i+1}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels: {labels[:5]}")
        custom_time = time.time() - start_time
        print(f"Custom loader time: {custom_time:.4f} seconds")

    except Exception as e:
        print(f"Error in test_custom_data_loader: {str(e)}")
        raise


def test_pytorch_data_loader():
    try:
        # Initialize PyTorch data loader
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        dataset = datasets.MNIST(
            os.path.join(os.path.dirname(__file__), "..", "data", "mnist"),
            train=True,
            download=True,
            transform=transform,
        )

        torch_loader = TorchDataLoader(
            dataset, batch_size=64, num_workers=4, shuffle=True
        )

        # Test a few batches
        print("\nTesting PyTorch data loader...")
        start_time = time.time()
        for i, (images, labels) in enumerate(torch_loader):
            if i >= 10:
                break
            print(f"Batch {i+1}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels: {labels[:5].numpy()}")
        torch_time = time.time() - start_time
        print(f"PyTorch loader time: {torch_time:.4f} seconds")

    except Exception as e:
        print(f"Error in test_pytorch_data_loader: {str(e)}")
        raise


if __name__ == "__main__":
    test_custom_data_loader()
    test_pytorch_data_loader()


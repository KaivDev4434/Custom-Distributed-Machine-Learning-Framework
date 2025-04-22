#include "DataLoader.h"
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <omp.h>

DataLoader::DataLoader(const std::string& data_path, int batch_size, int num_workers)
    : data_path_(data_path), batch_size_(batch_size), num_workers_(num_workers), current_idx_(0) {
    omp_set_num_threads(num_workers_);
}

DataLoader::~DataLoader() {}

void DataLoader::load_mnist(const std::string& images_file, const std::string& labels_file) {
    // Open files
    std::ifstream images(data_path_ + "/" + images_file, std::ios::binary);
    std::ifstream labels(data_path_ + "/" + labels_file, std::ios::binary);

    if (!images.is_open() || !labels.is_open()) {
        throw std::runtime_error("Failed to open MNIST files");
    }

    // Read MNIST header
    uint32_t magic_number, num_images, rows, cols;
    images.read(reinterpret_cast<char*>(&magic_number), 4);
    images.read(reinterpret_cast<char*>(&num_images), 4);
    images.read(reinterpret_cast<char*>(&rows), 4);
    images.read(reinterpret_cast<char*>(&cols), 4);

    // Convert from big endian to little endian
    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    // Read labels header
    uint32_t labels_magic, num_labels;
    labels.read(reinterpret_cast<char*>(&labels_magic), 4);
    labels.read(reinterpret_cast<char*>(&num_labels), 4);
    labels_magic = __builtin_bswap32(labels_magic);
    num_labels = __builtin_bswap32(num_labels);

    if (num_images != num_labels) {
        throw std::runtime_error("Number of images and labels don't match");
    }

    // Resize vectors
    images_.resize(num_images * rows * cols);
    labels_.resize(num_images);

    // Read data in parallel using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < num_images; ++i) {
        // Read image
        std::vector<unsigned char> image(rows * cols);
        images.read(reinterpret_cast<char*>(image.data()), rows * cols);
        
        // Read label
        unsigned char label;
        labels.read(reinterpret_cast<char*>(&label), 1);
        
        // Convert to float and normalize
        for (int j = 0; j < rows * cols; ++j) {
            images_[i * rows * cols + j] = static_cast<float>(image[j]) / 255.0f;
        }
        labels_[i] = static_cast<int>(label);
    }

    normalize_images();
    shuffle_data();
}

std::pair<std::vector<float>, std::vector<int>> DataLoader::get_next_batch() {
    if (current_idx_ >= images_.size() / (28 * 28)) {
        current_idx_ = 0;
        shuffle_data();
    }

    int batch_size = std::min(batch_size_, static_cast<int>(images_.size() / (28 * 28) - current_idx_));
    std::vector<float> batch_images(batch_size * 28 * 28);
    std::vector<int> batch_labels(batch_size);

    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        std::copy(images_.begin() + (current_idx_ + i) * 28 * 28,
                 images_.begin() + (current_idx_ + i + 1) * 28 * 28,
                 batch_images.begin() + i * 28 * 28);
        batch_labels[i] = labels_[current_idx_ + i];
    }

    current_idx_ += batch_size;
    return {batch_images, batch_labels};
}

size_t DataLoader::get_dataset_size() const {
    return images_.size() / (28 * 28);
}

int DataLoader::get_batch_size() const {
    return batch_size_;
}

void DataLoader::normalize_images() {
    // Images are already normalized during loading (divided by 255.0f)
}

void DataLoader::shuffle_data() {
    std::vector<size_t> indices(images_.size() / (28 * 28));
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<float> shuffled_images(images_.size());
    std::vector<int> shuffled_labels(labels_.size());

    #pragma omp parallel for
    for (size_t i = 0; i < indices.size(); ++i) {
        std::copy(images_.begin() + indices[i] * 28 * 28,
                 images_.begin() + (indices[i] + 1) * 28 * 28,
                 shuffled_images.begin() + i * 28 * 28);
        shuffled_labels[i] = labels_[indices[i]];
    }

    images_ = std::move(shuffled_images);
    labels_ = std::move(shuffled_labels);
} 
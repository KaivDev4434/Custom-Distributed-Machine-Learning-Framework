#pragma once

#include <vector>
#include <string>
#include <memory>
#include <mutex>

class DataLoader {
public:
    DataLoader(const std::string& data_path, int batch_size, int num_workers = 4);
    ~DataLoader();

    // Load MNIST data
    void load_mnist(const std::string& images_file, const std::string& labels_file);
    
    // Get next batch
    std::pair<std::vector<float>, std::vector<int>> get_next_batch();

    // Get dataset size
    size_t get_dataset_size() const;
    int get_batch_size() const;

private:
    std::string data_path_;
    int batch_size_;
    int num_workers_;
    size_t current_idx_;
    
    // Data storage
    std::vector<float> images_;
    std::vector<int> labels_;
    
    // Helper functions
    void normalize_images();
    void shuffle_data();
}; 
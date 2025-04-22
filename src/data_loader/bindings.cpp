#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "DataLoader.h"

namespace py = pybind11;

PYBIND11_MODULE(data_loader, m) {
    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<const std::string&, int, int>(),
             py::arg("data_path"),
             py::arg("batch_size"),
             py::arg("num_workers") = 4)
        .def("load_mnist", &DataLoader::load_mnist,
             py::arg("images_file"),
             py::arg("labels_file"))
        .def("get_next_batch", [](DataLoader& self) {
            auto [images, labels] = self.get_next_batch();
            
            // Create shape containers
            std::vector<py::ssize_t> image_shape = {
                static_cast<py::ssize_t>(self.get_batch_size()),
                28,
                28
            };
            std::vector<py::ssize_t> label_shape = {
                static_cast<py::ssize_t>(self.get_batch_size())
            };
            
            // Create numpy arrays with proper shape
            py::array_t<float> py_images(image_shape);
            py::array_t<int> py_labels(label_shape);
            
            auto images_buf = py_images.request();
            auto labels_buf = py_labels.request();
            
            float* images_ptr = static_cast<float*>(images_buf.ptr);
            int* labels_ptr = static_cast<int*>(labels_buf.ptr);
            
            std::copy(images.begin(), images.end(), images_ptr);
            std::copy(labels.begin(), labels.end(), labels_ptr);
            
            return std::make_pair(py_images, py_labels);
        })
        .def("get_dataset_size", &DataLoader::get_dataset_size)
        .def("get_batch_size", &DataLoader::get_batch_size);
} 
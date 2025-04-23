#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Model.h"

namespace py = pybind11;

PYBIND11_MODULE(nn_model, m) {
    py::class_<Model>(m, "Model")
        .def(py::init<int, int, int>(),
             py::arg("input_size"),
             py::arg("hidden_size"),
             py::arg("num_classes"))
        .def("forward", [](Model& self, py::array_t<float, py::array::c_style> input) {
            // Get input data
            auto buf = input.request();
            float* ptr = static_cast<float*>(buf.ptr);
            
            // Check dimensions
            if (buf.ndim != 2) {
                throw std::runtime_error("Input must be 2D array");
            }
            
            int batch_size = buf.shape[0];
            
            // Forward pass
            self.forward(ptr, batch_size);
            
            // Return success
            return true;
        }, py::arg("input"))
        .def("backward", [](Model& self, py::array_t<int, py::array::c_style> labels) {
            // Get label data
            auto buf = labels.request();
            int* ptr = static_cast<int*>(buf.ptr);
            
            // Check dimensions
            if (buf.ndim != 1) {
                throw std::runtime_error("Labels must be 1D array");
            }
            
            int batch_size = buf.shape[0];
            
            // Backward pass
            self.backward(ptr, batch_size);
            
            // Return success
            return true;
        }, py::arg("labels"))
        .def("get_output", [](Model& self) {
            // TODO: Need to know the batch size to properly transfer data
            return py::none();
        })
        .def("get_loss", &Model::get_loss)
        .def("update", &Model::update, py::arg("learning_rate") = 0.01)
        .def("zero_grad", &Model::zero_grad)
        .def("save", &Model::save, py::arg("filename"))
        .def("load", &Model::load, py::arg("filename"));
} 
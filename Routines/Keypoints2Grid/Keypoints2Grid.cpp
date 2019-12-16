#include <eigen3/Eigen/Dense>
#include <vector>
#include <cstdlib>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "LoaderVOX.h"

namespace py = pybind11;


void project_and_save(float val, py::array_t<float> &kps, std::string filename_cad, std::string customname_out) {
  assert(kps.ndim() == 2 && "Keypoints array must be 2D array!");
  assert(kps.shape(0) == 3 && "Keypoints array must have n_cols = 3!");
  const int n_kps = kps.shape(1);
  assert(n_kps > 0 && "Keypoints array must have n_rows > 0!");

  Eigen::MatrixXf kps1 = Eigen::Map<Eigen::MatrixXf>(kps.mutable_data(), 3, n_kps);

  Vox vox = load_vox(filename_cad);
  vox.pdf.resize(vox.dims(0)*vox.dims(1)*vox.dims(2));
  for (int i = 0; i < kps1.cols(); i++) {
    std::fill(vox.pdf.begin(), vox.pdf.end(), 0.0f);
    Eigen::Vector3i p = (vox.grid2world.inverse().eval()*Eigen::Vector4f(kps1(0, i), kps1(1, i), kps1(2, i), 1.0f)).topRows(3).array().round().cast<int>();
    if ((p.array() >= Eigen::Array3i(0, 0, 0)).all() && (p.array() < vox.dims.array()).all()) {
      vox.pdf[p(2)*vox.dims(1)*vox.dims(0) + p(1)*vox.dims(0) + p(0)] = val;

      std::string filename = customname_out + std::to_string(i) + std::string(".vox2");
      save_vox(filename, vox);
    }
  }
}

/**
 * Helper function for generating a heatmap CAD vox
 *
 * Currently each voxel has < 0.05 prob to be a keypoint
 *
 * For generating negative samples.
 */
void random_heatmap_and_save(float val, std::string filename_cad, std::string customname_out) {

  Vox vox = load_vox(filename_cad);
  vox.pdf.resize(vox.dims(0)*vox.dims(1)*vox.dims(2));
  float prob_t = 0.05;
  for (int i = 0; i < vox.dims[0]; ++i) {
    for (int j = 0; j < vox.dims[1]; ++j) {
      for (int k = 0; k < vox.dims[2]; ++k) {
        if (rand() % 1 < prob_t) {
          vox.pdf[k*vox.dims(1)*vox.dims(0) + j*vox.dims(0) + i] = 1;
        }
      }
    }
  }
  std::string filename = customname_out + "neg1" + std::string(".vox2");
  save_vox(filename, vox);
}


PYBIND11_MODULE(Keypoints2Grid, m) {
  m.def("project_and_save", &project_and_save, "project_and_save function");
  m.def("random_heatmap_and_save", &random_heatmap_and_save, "random_heatmap_and_save function");
}

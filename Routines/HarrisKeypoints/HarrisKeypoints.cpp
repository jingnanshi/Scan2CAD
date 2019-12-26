#include <vector>
#include <cstdlib>
#include <random>

#include <eigen3/Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>

#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/sift_keypoint.h>

#include <pcl/point_types.h>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

template <typename T>
typename pcl::PointCloud<T>::Ptr eigen2p_pcl(const Eigen::MatrixXf points){
  typename pcl::PointCloud<T>::Ptr p_pc(new typename pcl::PointCloud<T>());

  // Fill in the model cloud
  p_pc->width    = points.rows();
  p_pc->height   = 1;
  p_pc->is_dense = true;
  p_pc->points.resize (p_pc->width * p_pc->height);

  for (size_t i = 0; i < p_pc->points.size(); ++i){
    Eigen::Vector3f pt = points.row(i);
    p_pc->points[i].x = pt[0];
    p_pc->points[i].y = pt[1];
    p_pc->points[i].z = pt[2];
  }

  return p_pc;
}

template <typename T>
Eigen::MatrixXf p_pcl2eigen(const typename pcl::PointCloud<T>::Ptr p_pcl){
  Eigen::MatrixXf point_eigen(p_pcl->size(), 3);
  for(size_t i=0;i<p_pcl->size();++i){
    point_eigen(i, 0) = p_pcl->points[i].x;
    point_eigen(i, 1) = p_pcl->points[i].y;
    point_eigen(i, 2) = p_pcl->points[i].z;
  }
  return point_eigen;
}

/**
 * Helper function to run 3D Harris Keypoints detection on the given points
 * points: N-by-3 eigen matrix
 * radius: radius used in 3D Harris keypoint calculation
 * nms_threshold: non-maximal suppression threshold
 * threads: number of threads used in Harris keypoint calculation (0 is automatic)
 */
Eigen::MatrixXf get_harris_keypoints(
    py::array_t<float> &pypoints,
    const float radius,
    const float nms_threshold,
    const int threads) {

  assert(pypoints.ndim() == 2 && "Keypoints array must be 2D array!");
  assert(pypoints.shape(0) > 0 && "Keypoints array must have n_cols = 3!");
  assert(pypoints.shape(1) > 0 && "Keypoints array must have n_rows > 0!");
  std::cout << "pypoints shape:" << pypoints.shape(0) << std::endl;
  Eigen::MatrixXf points = Eigen::Map<Eigen::MatrixXf>(pypoints.mutable_data(), pypoints.shape(0), 3);

  pcl::PointCloud<pcl::PointXYZ>::Ptr p_pc = eigen2p_pcl<pcl::PointXYZ>(points);

  pcl::PointCloud<pcl::PointXYZI>::Ptr p_keypoints (new pcl::PointCloud<pcl::PointXYZI> ());
  pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> harris_detector;

  // run PCL harris 3d point detection
  harris_detector.setMethod(pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>::HARRIS);
  harris_detector.setNonMaxSupression (true);
  harris_detector.setRadius(radius);
  harris_detector.setRefine(true);
  harris_detector.setThreshold(nms_threshold);
  harris_detector.setNumberOfThreads(threads);
  harris_detector.setInputCloud(p_pc);
  harris_detector.compute(*p_keypoints);

  return p_pcl2eigen<pcl::PointXYZI>(p_keypoints);
}

PYBIND11_MODULE(HarrisKeypoints, m) {
  m.def("get_harris_keypoints", &get_harris_keypoints, "function get harris keypoints");
}

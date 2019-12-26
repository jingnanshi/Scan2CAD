all : HarrisKeypoints.so

HarrisKeypoints.so : HarrisKeypoints.cpp
	c++ -I../Base -I /usr/include/eigen3 -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` $^ -o $@ -lpcl_common -lpcl_io -lpcl_filters -lpcl_features -lpcl_keypoints -lpcl_registration -I /home/jnshi/.local/include/pcl-1.9/

clean:
	rm HarrisKeypoints.so

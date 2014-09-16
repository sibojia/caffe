#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <stdio.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/net.hpp"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;


int main(int argc, char** argv){
	Net<float> *net = new Net<float>("srcnn_train.prototxt");
	cout << net->layers().size() << '\n';
	vector<Blob<float>*> bottom_vec;
	net->ForwardBackward(bottom_vec);
	NetParameter p;
	net->ToProto(&p);
	WriteProtoToTextFile(p, "net.txt");
	cin.get();
}
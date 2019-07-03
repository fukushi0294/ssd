#include <iostream>

#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/util/command_line_flags.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/io/path.h>

#include <opencv2/highgui.hpp>

const std::string image_file = "C:/Users/kouji/Desktop/ssd_import/SSD-Tensorflow/demo/dog.jpg";

using namespace tensorflow;
using namespace tensorflow::ops;

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
	Tensor* output) {
	tensorflow::uint64 file_size = 0;
	TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

	string contents;
	contents.resize(file_size);

	std::unique_ptr<tensorflow::RandomAccessFile> file;
	TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

	tensorflow::StringPiece data;
	TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
	if (data.size() != file_size) {
		return tensorflow::errors::DataLoss("Truncated read of '", filename,
			"' expected ", file_size, " got ",
			data.size());
	}
	output->scalar<string>()() = string(data);
	return Status::OK();
}


Status ReadTensorFromImageFile(const string& file_name, const int input_height,
	const int input_width, const float input_mean,
	const float input_std,
	std::vector<Tensor>* out_tensors)
{
	auto root = tensorflow::Scope::NewRootScope();
	string input_name = "file_reader";
	string output_name = "normalized";
	Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());

	TF_RETURN_IF_ERROR(ReadEntireFile(tensorflow::Env::Default(), file_name, &input));
	auto file_reader =
		Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
		{ "input", input },
	};

	// Now try to figure out what kind of file it is and decode it.
	const int wanted_channels = 3;
	tensorflow::Output image_reader;
	if (tensorflow::str_util::EndsWith(file_name, ".png")) {
		image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
			DecodePng::Channels(wanted_channels));
	}else{
		// Assume if it's neither a PNG nor a GIF then it must be a JPEG.
		image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
			DecodeJpeg::Channels(wanted_channels));
	}
	// Now cast the image data to float so we can do normal math on it.
	auto float_caster =
		Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

	auto dims_expander = ExpandDims(root, float_caster, 0);
	auto resized = ResizeBilinear(
		root, dims_expander,
		Const(root.WithOpName("size"), { input_height, input_width }));
	// Subtract the mean and divide by the scale.
	Div(root.WithOpName(output_name), Sub(root, resized, { input_mean }),
	{ input_std });

	// This runs the GraphDef network definition that we've just constructed, and
	// returns the results in the output tensor.
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
		tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	TF_RETURN_IF_ERROR(session->Run({ inputs }, { output_name }, {}, out_tensors));

	return Status::OK();
}

Status ReadTensorFromMat(const string& file_name, const float input_mean,
	const float input_std,
	std::vector<Tensor>* out_tensors) {
	auto root = tensorflow::Scope::NewRootScope();
	string input_name = "file_reader";
	string output_name = "normalized";

	cv::Mat src = cv::imread(file_name);
	int32 input_width = src.cols;
	int32 input_height = src.rows;

	Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1,input_height,input_width,3 }));
	float *p = input_tensor.flat<float>().data();
	cv::Mat dest(input_height, input_width, CV_32FC3, p);
	src.convertTo(dest, CV_32FC3);

	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
		{ "input", input_tensor },
	};

	auto float_caster =
		Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_FLOAT);

	auto resized = ResizeBilinear(
		root, float_caster,
		Const(root.WithOpName("size"), { input_height, input_width }));

	// Subtract the mean and divide by the scale.
	Div(root.WithOpName(output_name), Sub(root, resized, { input_mean }),
	{ input_std });
	
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
		tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	TF_RETURN_IF_ERROR(session->Run({ inputs }, { output_name }, {}, out_tensors));

	return Status::OK();

}

Status LoadGraph(const string& graph_file_name,
	std::unique_ptr<tensorflow::Session>* session) {
	tensorflow::GraphDef graph_def;
	Status load_graph_status =
		ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
	if (!load_graph_status.ok()) {
		return tensorflow::errors::NotFound("Failed to load compute graph at '",
			graph_file_name, "'");
	}
	if (!graph_def.node_size()>0) {
		return tensorflow::errors::NotFound("Your graph :'",
			graph_file_name, "' has no node. Plese create freeze graph.");
	}

	session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
	Status session_create_status = (*session)->Create(graph_def);
	if (!session_create_status.ok()) {
		return session_create_status;
	}
	return Status::OK();
}



int main(int argc, char* argv[]) {
	tensorflow::port::InitMain(argv[0], &argc, &argv);

	string root_dir = "";
	string graph = "C:/Users/kouji/Desktop/ssd_import/export/trained_graph.pb";

	std::unique_ptr<tensorflow::Session> session;
	string graph_path = tensorflow::io::JoinPath(root_dir, graph);
	Status load_graph_status = LoadGraph(graph_path, &session);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << load_graph_status;
		return -1;
	}

	string image = "C:/Users/kouji/Desktop/ssd_import/SSD-Tensorflow/demo/dog.jpg";
	std::vector<Tensor> resized_tensors;
	string image_path = tensorflow::io::JoinPath(root_dir, image);

	int32 input_width = 576;
	int32 input_height = 768;
	float input_mean = 0;
	float input_std = 255;
	string input_layer = "input";
	string output_layer = "output";

	Status read_tensor_status =
		ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
			input_std, &resized_tensors);
	if (!read_tensor_status.ok()) {
		LOG(ERROR) << read_tensor_status;
		return -1;
	}
	const Tensor& resized_tensor = resized_tensors[0];

	std::vector<Tensor> cv_tensors;
	Status read_cv_status =
		ReadTensorFromMat(image_path, input_mean, input_std, &cv_tensors);
	if (!read_cv_status.ok()) {
		LOG(ERROR) << read_cv_status;
		return -1;
	}

	// Actually run the image through the model.
	std::vector<Tensor> outputs;
	Status run_status = session->Run({ { input_layer, resized_tensor } },
	{ output_layer }, {}, &outputs);
	if (!run_status.ok()) {
		LOG(ERROR) << "Running model failed: " << run_status;
		return -1;
	}

}
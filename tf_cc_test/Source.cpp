#include <iostream>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/util/command_line_flags.h>

using namespace tensorflow;

const string graph_filename = "./export/trained_graph.pb";
const string demo_dirname = "./SSD-Tensorflow/demo";
const string checkpointPath = "./SSD-Tensorflow/checkpoints/ssd_300_vgg.ckpt";

Status LoadGraph(string graph_filename,
	std::unique_ptr<GraphDef> &graph_def,
	std::unique_ptr<Session>& session)
{
	Status load_graph_status =
		ReadBinaryProto(Env::Default(), graph_filename, graph_def.get());
	if (!load_graph_status.ok()) {
		return errors::NotFound("Failed to load compute graph at '",
			graph_filename, "'");
	}
	session.reset(NewSession(SessionOptions()));

	Status session_create_status = session->Create(*graph_def);
	if (!session_create_status.ok()) {
		return session_create_status;
	}
	return Status::OK();
}

Tensor CreateTensorFromImage(string image_filename)
{


}


int main() {

	std::unique_ptr<Session> session;
	std::unique_ptr<GraphDef> graph(new GraphDef());

	auto load_graph_status = LoadGraph(graph_filename, graph, session);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << load_graph_status.error_message();
		return -1;
	}

	// Status load_graph_status = ReadBinaryProto(Env::Default(), graph_filename, graph.get());
	// if (!load_graph_status.ok()) {
	//     throw std::runtime_error("Error reading graph definition from " + graph_filename + ": " + load_graph_status.ToString());
	// }

	// session.reset(NewSession(SessionOptions()));
	// Status session_create_status = (session)->Create(*graph);
	// if (!session_create_status.ok()) {
	//     throw std::runtime_error("Error reading load graph : " + load_graph_status.ToString());
	// }

	/* If Graph is freeze graph, there is no need to refer checkpoint */
	// Tensor checkpointPathTensor(DT_STRING, TensorShape());
	// checkpointPathTensor.scalar<std::string>()() = checkpointPath;

	// auto session_run_status = session->Run(&input_tensor,
	//                                      {"accuracy:0"},
	//                                      {},
	//                                      &outputs);


}
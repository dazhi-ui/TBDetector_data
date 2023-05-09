
#include <fstream>
#include <pthread.h>
#include <sys/types.h>
#include "include/helper.hpp"
#include "include/def.hpp"
#include "include/histogram.hpp"
#include "../extern/extern.hpp"
#include "wl.hpp"
#include "graphchi_basic_includes.hpp"
#include "logger/logger.hpp"

using namespace graphchi;

graphchi_dynamicgraph_engine<VertexDataType, EdgeDataType> * dyngraph_engine;
std::string stream_file;
std::string sketch_file;
pthread_barrier_t std::graph_barrier;
pthread_barrier_t std::stream_barrier;
int std::stop = 0;
bool std::base_graph_constructed = false;
bool std::no_new_tasks = false;
int DECAY;
float LAMBDA;
int WINDOW;
int BATCH;
bool CHUNKIFY = true;
int CHUNK_SIZE;
FILE * SFP;
#ifdef VIZ
std::string HIST_FILE;
#endif

void * dynamic_graph_reader(void * info) {
#ifdef DEBUG
    logstream(LOG_DEBUG) << "Stream provenance graph from file: " << stream_file << std::endl;
#endif
    while(!std::base_graph_constructed) {
#ifdef DEBUG
        logstream(LOG_DEBUG) << "Waiting for the base graph to be constructed..." << std::endl;
#endif
        sleep(0);
    }
    Histogram* hist = Histogram::get_instance();
    hist->create_sketch();
    if (SFP == NULL)
	logstream(LOG_ERROR) << "Sketch file no longer exists..." << std::endl;
    assert(SFP != NULL);
#ifdef BASESKETCH
    for (int i = 0; i < SKETCH_SIZE; i++)
	fprintf(SFP,"%lu ", hist->get_sketch()[i]);
    fprintf(SFP, "\n");
#endif
    FILE * f = fopen(stream_file.c_str(), "r");
    if (f == NULL)
        logstream(LOG_ERROR) << "Unable to open the file for streaming: " << stream_file << ". Error code: " << strerror(errno) << std::endl;
    assert(f != NULL);

    vid_t srcID;
    vid_t dstID;
    EdgeDataType e;
    char s[1024];
    int cnt = 0;
    bool passed_barrier = false;

    while(fgets(s, 1024, f) != NULL) {
        if (cnt == 0 && !passed_barrier) {
            pthread_barrier_wait(&std::stream_barrier);
#ifndef USEWINDOW
	    for (int i = 0; i < SKETCH_SIZE; i++)
		fprintf(SFP,"%lu ", hist->get_sketch()[i]);
	    fprintf(SFP, "\n");
#ifdef VIZ
	    hist->write_histogram();
#endif
#endif
        }
        passed_barrier = true;
	FIXLINE(s);
        char delims[] = ":\t ";
        unsigned char *t;
        char *k;

        k = strtok(s, delims);
        if (k == NULL)
            logstream(LOG_ERROR) << "Source ID is missing." << std::endl;
        assert(k != NULL);
        srcID = atoi(k);

        k = strtok(NULL, delims);
        if (k == NULL)
            logstream(LOG_ERROR) << "Destination ID is missing." << std::endl;
        assert(k != NULL);
        dstID = atoi(k);
        e.itr = 0; 
        t = (unsigned char *)strtok(NULL, delims);
        if (t == NULL)
            logstream(LOG_ERROR) << "Source label is missing." << std::endl;
        assert(t != NULL);
        e.src[0] = hash(t);

        t = (unsigned char *)strtok(NULL, delims);
        if (t == NULL)
            logstream(LOG_ERROR) << "Destination label is missing." << std::endl;
        assert (t != NULL);
        e.dst = hash(t);

        t = (unsigned char *)strtok(NULL, delims);
        if (t == NULL)
            logstream(LOG_ERROR) << "Edge label is missing." << std::endl;
        assert (t != NULL);
        e.edg = hash(t);

        k = strtok(NULL, delims);
        if (k == NULL)
            logstream(LOG_ERROR) << "New_src info is missing." << std::endl;
        assert(k != NULL);
        int new_src = atoi(k);
        if (new_src == 1)
            e.new_src = true;
        else
            e.new_src = false;

        k = strtok(NULL, delims);
        if (k == NULL)
            logstream(LOG_ERROR) << "New_dst info is missing." << std::endl;
        assert(k != NULL);
        int new_dst = atoi(k);
        if (new_dst == 1)
            e.new_dst = true;
        else
            e.new_dst = false;

        k = strtok(NULL, delims);
        if (k == NULL)
            logstream(LOG_ERROR) << "Time is missing." << std::endl;
        assert (k != NULL);
        e.tme[0] = strtoul(k, NULL, 10);

#ifdef DEBUG
        k = strtok(NULL, delims);
        if (k != NULL)
            logstream(LOG_DEBUG) << "Extra info in the edge is ignored." << std::endl;
#endif
        if (srcID == dstID) {
#ifdef DEBUG
            logstream(LOG_ERROR) << "Ignore an edge because it is a self-loop: " << srcID << "<->" << dstID <<std::endl;
#endif
            continue;
        }
        bool success = false;
        while (!success)
            success = dyngraph_engine->add_edge(srcID, dstID, e);
        ++cnt;
        dyngraph_engine->add_task(srcID);
        dyngraph_engine->add_task(dstID);
#ifdef DEBUG
        logstream(LOG_DEBUG) << "Schedule a new edge: " << srcID << " -> " << dstID << std::endl;
#endif
        if (cnt == BATCH) {
            cnt = 0;
            passed_barrier = false;
            pthread_barrier_wait(&std::graph_barrier);
        }
    }
    std::stop = 1;
    if (cnt != 0) {
	    pthread_barrier_wait(&std::graph_barrier);
    }

    if (ferror(f) != 0 || fclose(f) != 0) {
        logstream(LOG_ERROR) << "Unable to close the stream file: " << stream_file << ". Error code: " << strerror(errno) << std::endl;
	return NULL;
    }
    
    return NULL;
}

int main(int argc, const char ** argv) {

    graphchi_init(argc, argv);
    metrics m("Streaming Extractor");
#ifdef DEBUG
    global_logger().set_log_level(LOG_DEBUG);
#else
    global_logger().set_log_level(LOG_INFO);
#endif
    std::string base_file = get_option_string("base");
    int niters = get_option_int("niters", 1000000);
    bool scheduler = true;	
    stream_file = get_option_string("stream");
    DECAY = get_option_int("decay", 10);
    LAMBDA = get_option_float("lambda", 0.02);	
    BATCH = get_option_int("batch", 1000);
    WINDOW = get_option_int("window", 500);
    sketch_file = get_option_string("sketch");
#ifdef VIZ
    HIST_FILE = get_option_string("histogram");
#endif
    int to_chunk = get_option_int("chunkify", 1);
    if (!to_chunk) CHUNKIFY = false;
    CHUNK_SIZE = get_option_int("chunk_size", 5);

    SFP = fopen(sketch_file.c_str(), "a");
    if (SFP == NULL) {
        logstream(LOG_ERROR) << "Cannot open the sketch file to write: " << sketch_file << ". Error code: " << strerror(errno) << std::endl;
    }
    assert(SFP != NULL);

    int nshards = convert_if_notexists<EdgeDataType>(base_file, get_option_string("nshards", "auto"));

    dyngraph_engine = new graphchi_dynamicgraph_engine<VertexDataType, EdgeDataType>(base_file, nshards, scheduler, m); 

    pthread_barrier_init(&std::stream_barrier, NULL, 2);
    pthread_barrier_init(&std::graph_barrier, NULL, 2);
    pthread_t strthread;
    int ret = pthread_create(&strthread, NULL, dynamic_graph_reader, NULL);
    assert(ret >= 0);

    WeisfeilerLehman program;
    dyngraph_engine->run(program, niters);

    Histogram* hist = Histogram::get_instance();
#ifdef DEBUG
    logstream(LOG_DEBUG) << "Recording the final graph sketch..." << std::endl;
#endif
    if (SFP == NULL)
        logstream(LOG_ERROR) << "Sketch file no longer exists..." << std::endl;
    assert(SFP != NULL);
    hist->record_sketch(SFP);
    if (ferror(SFP) != 0 || fclose(SFP) != 0) {
        logstream(LOG_ERROR) << "Unable to close the sketch file: " << sketch_file <<  std::endl;
        return -1;
    }
    int ret_stream = pthread_barrier_destroy(&std::stream_barrier);
    int ret_graph = pthread_barrier_destroy(&std::graph_barrier);
    if (ret_stream == EBUSY) {
        logstream(LOG_ERROR) << "Resource stream_barrier cannot be destroyed." << std::endl;
    }
    if (ret_graph == EBUSY) {
        logstream(LOG_ERROR) << "Resource graph_barrier cannot be destroyed." << std::endl;
    }
#ifdef DEBUG
    metrics_report(m);
#endif
    return 0;
}

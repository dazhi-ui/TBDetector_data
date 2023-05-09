
#include <thread>
#include <mutex>
#include <vector>
#include <sstream>
#include <string>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <pthread.h>
#include "graphchi_basic_includes.hpp"
#include "engine/dynamic_graphs/graphchi_dynamicgraph_engine.hpp"
#include "logger/logger.hpp"
#include "../extern/extern.hpp"
#include "include/def.hpp"
#include "include/helper.hpp"
#include "include/histogram.hpp"

namespace graphchi {
    struct WeisfeilerLehman : public GraphChiProgram<VertexDataType, EdgeDataType> {
        Histogram* hist = Histogram::get_instance();

        void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
#ifdef DEBUG
            if (vertex.num_edges() <= 0) {
	        logstream(LOG_DEBUG) << "Isolated vertex #"<< vertex.id() <<" detected." << std::endl;
		assert(false);
	    }
#endif
            if (gcontext.iteration == 0) {
		VertexDataType nl;

		if (vertex.num_inedges() > 0) {
		    graphchi_edge<EdgeDataType> * edge = vertex.inedge(0); 
		    nl.lb[0] = edge->get_data().dst;
		    nl.is_leaf = false;

		    for (int i = 0; i < vertex.num_inedges(); i++) {
			graphchi_edge<EdgeDataType> * in_edge = vertex.inedge(i);
			EdgeDataType el = in_edge->get_data();
			el.itr++; 
			in_edge->set_data(el);
		    }
		} else {
                    graphchi_edge<EdgeDataType> * edge = vertex.random_outedge();
                    nl.lb[0] = edge->get_data().src[0];
                    nl.is_leaf = true;
		}
		nl.tm[0] = 0; 
		vertex.set_data(nl);

		hist->update(nl.lb[0], true);

		if (gcontext.scheduler != NULL) {
		    gcontext.scheduler->add_task(vertex.id());
		}
#ifdef DEBUG
		logstream(LOG_DEBUG) << "Original Label (" << vertex.id() << "): " << nl.lb[0] << std::endl;
#endif
            } else if (gcontext.iteration < K_HOPS + 1){
#ifdef DEBUG
		for (int i = 0; i < vertex.num_outedges(); i++) {
		    graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
		    EdgeDataType el = out_edge->get_data();
		    if (el.itr == 0)
			assert(false);
		}
#endif

                std::vector<EdgeDataType> neighborhood; 
		for (int i = 0; i < vertex.num_inedges(); i++) {
                    graphchi_edge<EdgeDataType> * in_edge = vertex.inedge(i);
		    EdgeDataType el = in_edge->get_data();
		    assert(el.itr == gcontext.iteration);	
		    neighborhood.push_back(el);
		    el.itr++;
		    in_edge->set_data(el);
		}
		VertexDataType nl = vertex.get_data();

		if (neighborhood.size() == 0) {
		    unsigned long last_itr_label = nl.lb[gcontext.iteration - 1];
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "The label string of the base leaf vertex (" << vertex.id() << "): " << last_itr_label << std::endl;
#endif
		    hist->update(last_itr_label, true);
		    nl.lb[gcontext.iteration] = last_itr_label;
		    nl.tm[gcontext.iteration] = 0; 
		    vertex.set_data(nl);
		    for (int i = 0; i < vertex.num_outedges(); i++) {
			graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
			EdgeDataType el = out_edge->get_data();
			el.src[gcontext.iteration] = last_itr_label;
			el.tme[gcontext.iteration] = el.tme[gcontext.iteration - 1]; 
			out_edge->set_data(el);
		    }
		} else {
		    std::sort(neighborhood.begin(), neighborhood.end(), EdgeSorter(gcontext.iteration - 1));
		    std::string new_label_str = "";
		    std::string first_str;
		    std::stringstream first_out;
		    first_out << vertex.get_data().lb[gcontext.iteration - 1];
		    first_str = first_out.str();
		    new_label_str += first_str; 
		    for (std::vector<EdgeDataType>::iterator it = neighborhood.begin(); it != neighborhood.end(); ++it) {
			if (gcontext.iteration == 1) {
			    std::string edge_str;
			    std::stringstream edge_out;
			    edge_out << it->edg;
			    edge_str = edge_out.str();
			    new_label_str += " " + edge_str;
			}
			std::string node_str;
			std::stringstream node_out;
			node_out << it->src[gcontext.iteration - 1];
			node_str = node_out.str();
			new_label_str += " " + node_str;
		    }
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "New label string of vertex (" << vertex.id() << "): " << new_label_str << std::endl;
#endif
		    unsigned long new_label = hash((unsigned char *)new_label_str.c_str());
		    if (!CHUNKIFY) {
			hist->update(new_label, true);
		    } else {
			std::vector<unsigned long> to_insert = chunkify((unsigned char *)new_label_str.c_str(), CHUNK_SIZE);
			for (std::vector<unsigned long>::iterator ti = to_insert.begin(); ti != to_insert.end(); ++ti)
			    hist->update(*ti, true);
		    }
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "New label of vertex (" << vertex.id() << "): " << new_label << std::endl;
#endif
		    nl.lb[gcontext.iteration] = new_label;
		    nl.tm[gcontext.iteration] = neighborhood[0].tme[gcontext.iteration - 1];
		    vertex.set_data(nl);
		    for (int i = 0; i < vertex.num_outedges(); i++) {
			graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
			EdgeDataType el = out_edge->get_data();
			el.src[gcontext.iteration] = new_label;


			el.tme[gcontext.iteration] = neighborhood[0].tme[gcontext.iteration - 1];
			out_edge->set_data(el);
		    }
		}
		if (gcontext.scheduler != NULL) {
		    if (gcontext.iteration < K_HOPS)
			gcontext.scheduler->add_task(vertex.id());
	    } else {
		bool is_new = false;
		for (int i = 0; i < vertex.num_outedges(); i++) {
		    graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
		    EdgeDataType el = out_edge->get_data();
		    if (el.new_src)
			is_new = true;
		}
		if (!is_new) {
		    for (int i = 0; i < vertex.num_inedges(); i++) {
			graphchi_edge<EdgeDataType> * in_edge = vertex.inedge(i);
			EdgeDataType el = in_edge->get_data();
			if (el.new_dst)
			    is_new = true;
		    }
		}
		if (is_new) {
		    if (vertex.num_inedges() == 0) {
#ifdef DEBUG
			logstream(LOG_DEBUG) << "Processing new leaf vertex: " << vertex.id() << std::endl;
#endif
			graphchi_edge<EdgeDataType> * out_edge = vertex.random_outedge(); /* The node must have at least one outedge. */
			assert(out_edge != NULL);
			EdgeDataType el = out_edge->get_data();
			VertexDataType nl;
			nl.lb[0] = el.src[0];
			nl.tm[0] = 0;
			for (int i = 1; i < K_HOPS + 1; i++) {
			    nl.lb[i] = nl.lb[0];
			    nl.tm[i] = 0;
			}
			nl.is_leaf = true;
			vertex.set_data(nl);
			for (int i = 0; i < K_HOPS + 1; i++) {
			    hist->decay(SFP);
			    hist->update(nl.lb[i], false);
			}
			for (int i = 0; i < vertex.num_outedges(); i++) {
			    graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
			    EdgeDataType el = out_edge->get_data();
			    for (int j = 1; j < K_HOPS + 1; j++) {
				el.src[j] = nl.lb[j];
				el.tme[j] = el.tme[j - 1];
			    }
			    el.new_src = false;
			    out_edge->set_data(el);
			}
			return;
		    } else {
#ifdef DEBUG
			logstream(LOG_DEBUG) << "Process new non-leaf vertex: " << vertex.id() << std::endl;
#endif
			graphchi_edge<EdgeDataType> * edge = vertex.inedge(0);
			VertexDataType nl = vertex.get_data();
			nl.lb[0] = edge->get_data().dst;
			nl.tm[0] = 0;
			vertex.set_data(nl);
			
			for (int i = 0; i < vertex.num_inedges(); i++) {
			    graphchi_edge<EdgeDataType> * in_edge = vertex.inedge(i);
			    EdgeDataType el = in_edge->get_data();
			    assert(el.itr == 0);
			    el.itr++; 
			    el.new_dst = false; 
			    in_edge->set_data(el);
			}

			for (int i = 0; i < vertex.num_outedges(); i++) {
			    graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
			    EdgeDataType el = out_edge->get_data();
			    el.new_src = false; 
			    out_edge->set_data(el);
			}
#ifdef DEBUG
			logstream(LOG_DEBUG) << "Vertex (" << vertex.id() << ") label: " << nl.lb[0] << std::endl;
#endif
			hist->decay(SFP);
			hist->update(nl.lb[0], false);
		    }
		}
		if (vertex.num_inedges() == 0) {
		    VertexDataType nl = vertex.get_data();
		    assert(nl.is_leaf); 
		    for (int i = 0; i < vertex.num_outedges(); i++) {
			graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
			EdgeDataType el = out_edge->get_data();
			for (int j = 1; j < K_HOPS + 1; j++) {
			    el.src[j] = nl.lb[j];
			    el.tme[j] = el.tme[j - 1];
			}
			out_edge->set_data(el);
		    }
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "Streaming refreshes an existing leaf node: " << vertex.id() << std::endl;
#endif
		} else {
		    VertexDataType nl = vertex.get_data();
		    if (nl.is_leaf)
			nl.is_leaf = false;
		    for (int i = 0; i < vertex.num_outedges(); i++) {
			graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
			EdgeDataType el = out_edge->get_data();
			for (int j = 1; j < K_HOPS + 1; j++) {
			    el.src[j] = nl.lb[j];
			    el.tme[j] = nl.tm[j];
			}
			out_edge->set_data(el);
		    }
		    int min_itr = K_HOPS + 2;
		    for (int i = 0; i < vertex.num_inedges(); i++) {
			graphchi_edge<EdgeDataType> * in_edge = vertex.inedge(i);
			EdgeDataType el = in_edge->get_data();
			if (el.itr == 0) {
			    el.itr++;
			    in_edge->set_data(el);
			}
			if (el.itr < min_itr)
			    min_itr = el.itr;
		    }
		    assert(min_itr > 0 && min_itr < K_HOPS + 2);
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "The min_itr of the vertex (" << vertex.id() << ") is: " << min_itr << std::endl;
#endif
		    if (min_itr == K_HOPS + 1)
			return;
		    std::vector<EdgeDataType> neighborhood;
		    for (int i = 0; i < vertex.num_inedges(); i++) {
			graphchi_edge<EdgeDataType> * in_edge = vertex.inedge(i);
			EdgeDataType el = in_edge->get_data();
			neighborhood.push_back(el);
			if (el.itr < K_HOPS + 1)
			    el.itr++; 
			in_edge->set_data(el);
		    }
		    std::sort(neighborhood.begin(), neighborhood.end(), EdgeSorter(min_itr - 1));
		    std::string new_label_str = "";
		    std::string first_str;
		    std::stringstream first_out;
		    first_out << vertex.get_data().lb[min_itr - 1];
		    first_str = first_out.str();
		    new_label_str += first_str; 

		    for (std::vector<EdgeDataType>::iterator it = neighborhood.begin(); it != neighborhood.end(); ++it) {
			if (min_itr == 1) { 
			    std::string edge_str;
			    std::stringstream edge_out;
			    edge_out << it->edg;
			    edge_str = edge_out.str();
			    new_label_str += " " + edge_str;
			}
			std::string node_str;
			std::stringstream node_out;
			node_out << it->src[min_itr - 1];
			node_str = node_out.str();
			new_label_str += " " + node_str;
		    }
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "New label string of the vertex (" << vertex.id() << "): " << new_label_str << std::endl;
#endif
		    unsigned long new_label = hash((unsigned char *)new_label_str.c_str());
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "New label of the vertex (" << vertex.id() << "): " << new_label << std::endl;
#endif
		    if (!CHUNKIFY) {
			hist->decay(SFP);
			hist->update(new_label, false);
		    } else {
			std::vector<unsigned long> to_insert = chunkify((unsigned char *)new_label_str.c_str(), CHUNK_SIZE);
			bool first = true;
			for (std::vector<unsigned long>::iterator ti = to_insert.begin(); ti != to_insert.end(); ++ti) {
			    if (first) {
				hist->decay(SFP);  
				first = false;
			    }
			    hist->update(*ti, false);
			}
		    
		    nl.lb[min_itr] = new_label;
		    vertex.set_data(nl);
		    for (int i = 0; i < vertex.num_outedges(); i++) {
			graphchi_edge<EdgeDataType> * out_edge = vertex.outedge(i);
			EdgeDataType el = out_edge->get_data();
			el.src[min_itr] = new_label;
			el.tme[min_itr] = neighborhood[0].tme[min_itr - 1];
#ifdef DEBUG
			logstream(LOG_DEBUG) << "Outgoing vertex (" << out_edge->vertex_id() << ") current itr:" << el.itr << std::endl;
#endif
			if (el.itr == K_HOPS + 1) {
			    el.itr = min_itr + 1;
#ifdef DEBUG
			    logstream(LOG_DEBUG) << "Update outgoing vertex (" << out_edge->vertex_id() << ") itr to: " << el.itr << std::endl;
#endif
			}
			out_edge->set_data(el);
			
			if (min_itr < K_HOPS) {
			    if (gcontext.scheduler != NULL)
				gcontext.scheduler->add_task(out_edge->vertex_id());			
			}
		    }
		    if (min_itr < K_HOPS + 1) {
			if (gcontext.scheduler != NULL)
			    gcontext.scheduler->add_task(vertex.id());
		    }
		}
	    }
	}
	
	void before_iteration(int iteration, graphchi_context &gcontext) {
	}

	void after_iteration(int iteration, graphchi_context &gcontext) {
#ifdef DEBUG
	    logstream(LOG_DEBUG) << "Current iteration: " << iteration << std::endl;
#endif
	    if (iteration == K_HOPS)
		std::base_graph_constructed = true;
	    if (std::no_new_tasks){
#ifdef DEBUG
		logstream(LOG_DEBUG) << "No new task at the moment...Let's see if we need to stop or wait..." << std::endl;
#endif
		if (std::stop) {
#ifdef DEBUG
		    logstream(LOG_DEBUG) << "Everything is done!" << std::endl;
#endif
		    gcontext.set_last_iteration(iteration); 
		    return;
		}
		pthread_barrier_wait(&std::stream_barrier);
		std::no_new_tasks = false;
#ifdef DEBUG
		logstream(LOG_DEBUG) << "No new tasks to run! But we have new streaming edges..." << std::endl;
#endif
		pthread_barrier_wait(&std::graph_barrier);
	    }
	}
	void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {
	}
	void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {
	}

    };
}

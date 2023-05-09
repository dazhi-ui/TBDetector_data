#ifndef __DEF_HPP__
#define __DEF_HPP__

#define _USE_MATH_DEFINES

#include <random>
#include <cmath>
#include <pthread.h> 
#include <string>

extern int DECAY;
extern float LAMBDA;
extern int WINDOW;
extern int BATCH;
extern bool CHUNKIFY;
extern int CHUNK_SIZE;
extern FILE * SFP;
#ifdef VIZ
extern std::string HIST_FILE;
#endif
typedef struct edge_label {
    unsigned long src[K_HOPS+1];
    unsigned long tme[K_HOPS+1];
    unsigned long dst;
    unsigned long edg;
    int itr;
    bool new_src;
    bool new_dst;
} EdgeDataType;

typedef struct node_label {
    unsigned long lb[K_HOPS+1];
    unsigned long tm[K_HOPS+1];
    bool is_leaf;
} VertexDataType;
struct hist_elem {
    double r[SKETCH_SIZE];
    double beta[SKETCH_SIZE];
    double c[SKETCH_SIZE]; 
};
std::gamma_distribution<double> gamma_dist(2.0, 1.0);
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

#endif /* __DEF_HPP__ */

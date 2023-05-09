#ifndef __HISTOGRAM_HPP__
#define __HISTOGRAM_HPP__

#include <iostream>
#include <map>
#include <vector>
#include <thread>
#include <mutex>
#include <math.h>
#include "logger/logger.hpp"
#include "def.hpp"
class Histogram {
public:
    static Histogram* get_instance();
    ~Histogram();
    struct hist_elem construct_hist_elem(unsigned long label);
    void decay(FILE* fp);
    void update(unsigned long label, bool base);
    void create_sketch();
    void record_sketch(FILE* fp);
    unsigned long* get_sketch();
#ifdef VIZ
    void write_histogram();
#endif
#ifdef DEBUG
    void print_histogram();
#endif

private:
    static Histogram* histogram;

    Histogram() {
        this->t = 0;
#ifdef USEWINDOW
        this->w = 0;
#endif
#ifdef VIZ
	this->c = 0;
#endif
        this->powerful = pow(M_E, -LAMBDA);
    }

    std::map<unsigned long, double> histogram_map; /* histogram_map maps a label to its value. */
    unsigned long sketch[SKETCH_SIZE];
    double hash[SKETCH_SIZE];
    double powerful;
#ifdef MEMORY   
    double gamma_param[PREGEN][SKETCH_SIZE];
    double r_beta_param[PREGEN][SKETCH_SIZE];
    double power_r[PREGEN][SKETCH_SIZE];
#endif

    int t; 
#ifdef USEWINDOW
    int w; 

#endif
#ifdef VIZ
    int c; 
#endif
    
    std::mutex histogram_map_lock;
};

#include "histogram.cpp"

#endif /* __HISTOGRAM_HPP__ */

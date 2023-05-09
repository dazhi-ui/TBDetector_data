#ifndef __HELPER_HPP__
#define __HELPER_HPP__

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>
#include "logger/logger.hpp"
#include "def.hpp"

namespace graphchi {
    unsigned long hash(unsigned char *str) {
        unsigned long hash = 5381;
        int c;

        while (c = *str++)
            hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
        return hash;
    }

    void parse(EdgeDataType &e, const char *s) {
        char *ss = (char *) s;
        char delims[] = ":";
        unsigned char *t;
        char *k;

        e.itr = 0;
        e.new_src = false;
        e.new_dst = false;
	
        t = (unsigned char *)strtok(ss, delims);
        if (t == NULL)
            logstream(LOG_ERROR) << "Source label is missing." << std::endl;
        assert(t != NULL);
        e.src[0] = hash(t);

        t = (unsigned char *)strtok(NULL, delims);
        if (t == NULL)
            logstream(LOG_ERROR) << "Destination label does is missing." << std::endl;
        assert (t != NULL);
        e.dst = hash(t);

        t = (unsigned char *)strtok(NULL, delims);
        if (t == NULL)
            logstream(LOG_ERROR) << "Edge label is missing." << std::endl;
        assert (t != NULL);
        e.edg = hash(t);

        k = strtok(NULL, delims);
        if (k == NULL)
            logstream(LOG_ERROR) << "Timestamp is missing." << std::endl;
        assert (k != NULL);
        e.tme[0] = std::strtoul(k, NULL, 10);
#ifdef DEBUG
        k = strtok(NULL, delims);
        if (k != NULL)
            logstream(LOG_ERROR) << "Extra info is ignored." << std::endl;
#endif

        return;
    }

    std::vector<unsigned long> chunkify(unsigned char *s, int chunk_size) {
        char *ss = (char *) s;
        char delims[] = " ";
        char *t;
        int counter = 0;
        std::string to_hash = "";
        std::vector<unsigned long> rtn;
        bool first = true;

        /* chunk_size must be larger than 1. */
        assert(chunk_size > 1);

        t = strtok(ss, delims);
        if (t == NULL)
            logstream(LOG_ERROR) << "The string to be chunked must be non-empty." << std::endl;
        assert(t != NULL);

        while (t != NULL) {
            std::string str(t);
            if (first) {
                to_hash += str;
                first = false;
            } else
                to_hash += " " + str;
            counter++;
            if (counter == chunk_size) {
                rtn.push_back(hash((unsigned char *)to_hash.c_str()));
                counter = 0;
                to_hash = "";
            }
            t = strtok(NULL, delims);
        }
        /* Handle the leftover (last substring) that might be < chunk_size. */
        if (to_hash.length() > 0) {
            rtn.push_back(hash((unsigned char *)to_hash.c_str()));
        }
        return rtn;
    }

    bool compareEdges(EdgeDataType a, EdgeDataType b, int pos) {
        return a.tme[pos] < b.tme[pos];
    }
    

    class EdgeSorter{
        int pos;

        public:
            EdgeSorter(int pos) {
                this->pos = pos;
            }

            bool operator()(EdgeDataType a, EdgeDataType b) const {
                return compareEdges(a, b, pos);
            }
    };

}

#endif /* __HELPER_HPP__ */

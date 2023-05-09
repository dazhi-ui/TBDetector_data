#ifndef storage_hpp
#define storage_hpp

#include "def.hpp"
#include "kissdb.h"

class Storage {
	public:
		/* Open a storage file. */
		virtual int init_storage(const char *path) = 0;
		virtual int get_item(const unsigned long *key, struct hist_elem *value) = 0;
		virtual int put_item(const unsigned long *key, struct hist_elem *value) = 0;
		virtual void close_storage() = 0;
};

class kiss_storage: public Storage {
	public:
		int init_storage(const char *path);
		int get_item(const unsigned long *key, struct hist_elem *value);
		int put_item(const unsigned long *key, struct hist_elem *value);
		void close_storage();

	private:
		KISSDB db;
};

extern kiss_storage ks;

#include "storage.cpp"

#endif

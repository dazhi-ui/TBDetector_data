
#ifndef ___KISSDB_H
#define ___KISSDB_H

#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KISSDB_VERSION 2

typedef struct {
	unsigned long hash_table_size;
	unsigned long key_size;
	unsigned long value_size;
	unsigned long hash_table_size_bytes;
	unsigned long num_hash_tables;
	uint64_t *hash_tables;
	FILE *f;
} KISSDB;
#define KISSDB_ERROR_IO -1

/**
 * Out of memory
 */
#define KISSDB_ERROR_MALLOC -2

/**
 * Invalid paramters (e.g. missing _size paramters on init to create database)
 */
#define KISSDB_ERROR_INVALID_PARAMETERS -3

/**
 * Database file appears corrupt
 */
#define KISSDB_ERROR_CORRUPT_DBFILE -4

/**
 * Open mode: read only
 */
#define KISSDB_OPEN_MODE_RDONLY 1

/**
 * Open mode: read/write
 */
#define KISSDB_OPEN_MODE_RDWR 2

/**
 * Open mode: read/write, create if doesn't exist
 */
#define KISSDB_OPEN_MODE_RWCREAT 3

/**
 * Open mode: truncate database, open for reading and writing
 */
#define KISSDB_OPEN_MODE_RWREPLACE 4

extern int KISSDB_open(
	KISSDB *db,
	const char *path,
	int mode,
	unsigned long hash_table_size,
	unsigned long key_size,
	unsigned long value_size);

/**
 * Close database
 *
 * @param db Database struct
 */
extern void KISSDB_close(KISSDB *db);

/**
 * Get an entry
 *
 * @param db Database struct
 * @param key Key (key_size bytes)
 * @param vbuf Value buffer (value_size bytes capacity)
 * @return -1 on I/O error, 0 on success, 1 on not found
 */
extern int KISSDB_get(KISSDB *db,const void *key,void *vbuf);

/**
 * Put an entry (overwriting it if it already exists)
 *
 * In the already-exists case the size of the database file does not
 * change.
 *
 * @param db Database struct
 * @param key Key (key_size bytes)
 * @param value Value (value_size bytes)
 * @return -1 on I/O error, 0 on success
 */
extern int KISSDB_put(KISSDB *db,const void *key,const void *value);

/**
 * Cursor used for iterating over all entries in database
 */
typedef struct {
	KISSDB *db;
	unsigned long h_no;
	unsigned long h_idx;
} KISSDB_Iterator;

/**
 * Initialize an iterator
 *
 * @param db Database struct
 * @param i Iterator to initialize
 */
extern void KISSDB_Iterator_init(KISSDB *db,KISSDB_Iterator *dbi);

/**
 * Get the next entry
 *
 * The order of entries returned by iterator is undefined. It depends on
 * how keys hash.
 *
 * @param Database iterator
 * @param kbuf Buffer to fill with next key (key_size bytes)
 * @param vbuf Buffer to fill with next value (value_size bytes)
 * @return 0 if there are no more entries, negative on error, positive if an kbuf/vbuf have been filled
 */
extern int KISSDB_Iterator_next(KISSDB_Iterator *dbi,void *kbuf,void *vbuf);

#ifdef __cplusplus
}
#endif

#include "../kissdb.cpp"

#endif
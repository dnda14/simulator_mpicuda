#ifndef DISTRIBUTED_HASH_TABLE_HPP
#define DISTRIBUTED_HASH_TABLE_HPP

#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstring>
#include <functional> 
#include <string>
#include <cstddef> // Para offsetof

// === 1. Estructuras de Datos ===

struct SimulationParams {
    int grid_x = 500;
    int grid_y = 1500;
    int num_species = 5;
    double dt = 0.1;
    int steps = 1000;
};

struct GridCell {
    double concentrations[5]; 
    double flux_in = 0.0;
    double flux_out = 0.0;

    GridCell() {
        for(int i=0; i<5; i++) concentrations[i] = 0.0;
    }
};

struct DHT_Bucket {
    int key;              
    GridCell value;       
    int status;           // 0 = Vacío, 1 = Ocupado
    unsigned int checksum; 
};

// === 2. Clase Base Distribuida ===

class DistributedHashTable {
protected:
    MPI_Win win;                 
    DHT_Bucket* local_buffer;    
    int rank, size;
    size_t local_capacity;       
    
public:
    DistributedHashTable(int total_expected_entries, int rank, int size) 
        : rank(rank), size(size) {
        
        local_capacity = (total_expected_entries / size) * 2;
        if (local_capacity < 100) local_capacity = 100;

        MPI_Alloc_mem(local_capacity * sizeof(DHT_Bucket), MPI_INFO_NULL, &local_buffer);
        memset(local_buffer, 0, local_capacity * sizeof(DHT_Bucket));

        // <--- CAMBIO IMPORTANTE AQUI ABAJO --->
        // Cambiamos el disp_unit de sizeof(DHT_Bucket) a 1.
        // Esto significa que MPI tratará los desplazamientos como BYTES,
        // coincidiendo con los cálculos que hacemos en las subclases.
        MPI_Win_create(local_buffer, 
                       local_capacity * sizeof(DHT_Bucket), 
                       1,              // <--- ESTE ERA EL ERROR (Antes era sizeof...)
                       MPI_INFO_NULL, 
                       MPI_COMM_WORLD, 
                       &win);
    }

    virtual ~DistributedHashTable() {
        MPI_Win_free(&win);
        MPI_Free_mem(local_buffer);
    }

    int getOwnerRank(int key) const {
        // Distribución simple: cada proceso obtiene un bloque contiguo de celdas
        return key % size;
    }

    size_t getLocalOffset(int key) const {
        // El offset local es simplemente key / size
        // Esto garantiza que el offset siempre esté dentro de local_capacity
        size_t offset = static_cast<size_t>(key) / size;
        if (offset >= local_capacity) {
            // Esto no debería pasar si local_capacity está bien calculado,
            // pero por seguridad hacemos wrap-around
            offset = offset % local_capacity;
        }
        return offset;
    }

    virtual void updateCell(int key, const GridCell& val) = 0;
    virtual GridCell getCell(int key) = 0;
    virtual std::string getStrategyName() const = 0;

    virtual void advectStep() {
        // Implementación vacía para benchmark
    }

    virtual void syncGhostCells() {
        MPI_Win_flush_all(win); 
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

#endif // DISTRIBUTED_HASH_TABLE_HPP
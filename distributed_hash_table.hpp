#ifndef DISTRIBUTED_HASH_TABLE_HPP
#define DISTRIBUTED_HASH_TABLE_HPP

#include <mpi.h>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <memory>
#include <iostream>

// Parámetros de la simulación
struct SimulationParams {
    int grid_x = 500;
    int grid_y = 1500;
    int num_species = 5;
    double dt = 0.1;
    double dx = 1.0;
    double velocity = 1.0;
    int steps = 1000;
};

// Estructura para una celda de la malla
struct GridCell {
    std::vector<double> concentrations;
    double flux_in = 0.0;
    double flux_out = 0.0;

    // ✅ Constructor por defecto (necesario para std::unordered_map::operator[])
    GridCell() = default;

    // ✅ Constructor con número de especies
    GridCell(int num_species) : concentrations(num_species, 0.0) {}
};

// Interfaz base para la tabla hash distribuida
class DistributedHashTable {
public:
    virtual ~DistributedHashTable() = default;

    virtual void updateCell(int cell_id, const GridCell& new_data) = 0;
    virtual GridCell getCell(int cell_id) = 0;
    virtual void advectStep() = 0;
    virtual void syncGhostCells() = 0;
    virtual std::string getStrategyName() const = 0;
};

#endif // DISTRIBUTED_HASH_TABLE_HPP

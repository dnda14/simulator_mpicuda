#ifndef COARSE_GRAINED_HASH_TABLE_HPP
#define COARSE_GRAINED_HASH_TABLE_HPP

#include <unordered_map>
#include <shared_mutex>
#include <memory>
#include <mpi.h>
#include "distributed_hash_table.hpp"  // Contiene GridCell y DistributedHashTable

// Implementación con bloqueo grueso (Coarse-Grained Locking)
class CoarseGrainedHashTable : public DistributedHashTable {
private:
    std::unordered_map<int, GridCell> local_data;
    std::shared_mutex global_mutex;
    int rank, size;
    int num_species;
    int local_grid_size;

public:
    CoarseGrainedHashTable(int num_species, int total_cells, int rank, int size)
        : num_species(num_species), rank(rank), size(size)
    {
        local_grid_size = total_cells / size;

        std::unique_lock lock(global_mutex);
        for (int i = 0; i < local_grid_size; ++i) {
            int global_id = rank * local_grid_size + i;
            local_data[global_id] = GridCell(num_species);
        }
    }

    void updateCell(int cell_id, const GridCell& new_data) override {
        std::unique_lock lock(global_mutex);
        auto it = local_data.find(cell_id);
        if (it != local_data.end()) {
            it->second = new_data;
        }
    }

    GridCell getCell(int cell_id) override {
        std::shared_lock lock(global_mutex);
        auto it = local_data.find(cell_id);
        if (it != local_data.end()) {
            return it->second;
        }
        return GridCell(num_species);
    }

    void advectStep() override {
        std::unique_lock lock(global_mutex);

        // Copia temporal para evitar inconsistencias durante la iteración
        auto temp_data = local_data;

        for (auto& [cell_id, cell] : local_data) {
            int left_cell = cell_id - 1;

            auto it_left = temp_data.find(left_cell);
            if (it_left != temp_data.end()) {
                const auto& left_data = it_left->second;

                for (int i = 0; i < num_species; ++i) {
                    cell.concentrations[i] +=
                        (left_data.concentrations[i] - cell.concentrations[i]);
                }
            }
        }
    }

    void syncGhostCells() override {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    std::string getStrategyName() const override {
        return "Coarse-Grained Locking";
    }
};

#endif // COARSE_GRAINED_HASH_TABLE_HPP

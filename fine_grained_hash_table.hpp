#ifndef FINE_GRAINED_HASH_TABLE_HPP
#define FINE_GRAINED_HASH_TABLE_HPP

#include <unordered_map>
#include <memory>
#include <vector>
#include <shared_mutex>
#include <algorithm>
#include <mpi.h>
#include "distributed_hash_table.hpp"  // Incluye la interfaz base y GridCell

// Implementación con bloqueo fino (Fine-Grained Locking)
class FineGrainedHashTable : public DistributedHashTable {
private:
    struct CellWithMutex {
        GridCell cell;
        std::shared_mutex mutex;

        CellWithMutex(int num_species) : cell(num_species) {}
    };

    std::unordered_map<int, std::unique_ptr<CellWithMutex>> local_data;
    int rank, size;
    int num_species;
    int local_grid_size;

public:
    FineGrainedHashTable(int num_species, int total_cells, int rank, int size)
        : num_species(num_species), rank(rank), size(size) 
    {
        local_grid_size = total_cells / size;

        for (int i = 0; i < local_grid_size; ++i) {
            int global_id = rank * local_grid_size + i;
            local_data[global_id] = std::make_unique<CellWithMutex>(num_species);
        }
    }

    void updateCell(int cell_id, const GridCell& new_data) override {
        auto it = local_data.find(cell_id);
        if (it != local_data.end()) {
            std::unique_lock lock(it->second->mutex);
            it->second->cell = new_data;
        }
    }

    GridCell getCell(int cell_id) override {
        auto it = local_data.find(cell_id);
        if (it != local_data.end()) {
            std::shared_lock lock(it->second->mutex);
            return it->second->cell;
        }
        return GridCell(num_species);
    }

    void advectStep() override {
        // Ordenar para prevenir deadlocks
        std::vector<int> cell_ids;
        cell_ids.reserve(local_data.size());
        for (const auto& pair : local_data) {
            cell_ids.push_back(pair.first);
        }
        std::sort(cell_ids.begin(), cell_ids.end());

        for (int cell_id : cell_ids) {
            int left_cell = cell_id - 1;

            auto it_current = local_data.find(cell_id);
            auto it_left = local_data.find(left_cell);
            if (it_left != local_data.end() && it_current != local_data.end()) {
                auto& current_mutex = it_current->second->mutex;
                auto& left_mutex = it_left->second->mutex;

                std::unique_lock lock1(current_mutex, std::defer_lock);
                std::shared_lock lock2(left_mutex, std::defer_lock);
                std::lock(lock1, lock2);

                auto& current_cell = it_current->second->cell;
                const auto& left_cell_data = it_left->second->cell;

                for (int i = 0; i < num_species; ++i) {
                    current_cell.concentrations[i] +=
                        (left_cell_data.concentrations[i] -
                         current_cell.concentrations[i]);
                }
            }
        }
    }

    void syncGhostCells() override {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    std::string getStrategyName() const override {
        return "Fine-Grained Locking";
    }
};

#endif // FINE_GRAINED_HASH_TABLE_HPP

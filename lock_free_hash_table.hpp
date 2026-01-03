#ifndef LOCK_FREE_HASH_TABLE_HPP
#define LOCK_FREE_HASH_TABLE_HPP

#include <unordered_map>
#include <atomic>
#include <memory>
#include <mpi.h>
#include "distributed_hash_table.hpp"  // Contiene GridCell y DistributedHashTable

// Implementación Lock-Free (sin bloqueos explícitos)
class LockFreeHashTable : public DistributedHashTable {
private:
    struct AtomicCell {
        std::atomic<double>* concentrations;
        std::atomic<double> flux_in;
        std::atomic<double> flux_out;
        int num_species;

        AtomicCell(int num_species) : num_species(num_species) {
            concentrations = new std::atomic<double>[num_species];
            for (int i = 0; i < num_species; ++i) {
                concentrations[i].store(0.0);
            }
            flux_in.store(0.0);
            flux_out.store(0.0);
        }

        ~AtomicCell() {
            delete[] concentrations;
        }
    };

    std::unordered_map<int, std::unique_ptr<AtomicCell>> local_data;
    int rank, size;
    int num_species;
    int local_grid_size;

public:
    LockFreeHashTable(int num_species, int total_cells, int rank, int size)
        : num_species(num_species), rank(rank), size(size)
    {
        local_grid_size = total_cells / size;

        // Inicializa celdas locales
        for (int i = 0; i < local_grid_size; ++i) {
            int global_id = rank * local_grid_size + i;
            local_data[global_id] = std::make_unique<AtomicCell>(num_species);
        }
    }

    void updateCell(int cell_id, const GridCell& new_data) override {
        auto it = local_data.find(cell_id);
        if (it != local_data.end()) {
            auto& cell = it->second;

            for (int i = 0; i < num_species; ++i) {
                cell->concentrations[i].store(new_data.concentrations[i],
                                              std::memory_order_relaxed);
            }
            cell->flux_in.store(new_data.flux_in, std::memory_order_relaxed);
            cell->flux_out.store(new_data.flux_out, std::memory_order_relaxed);
        }
    }

    GridCell getCell(int cell_id) override {
        GridCell result(num_species);

        auto it = local_data.find(cell_id);
        if (it != local_data.end()) {
            auto& cell = it->second;

            for (int i = 0; i < num_species; ++i) {
                result.concentrations[i] = cell->concentrations[i].load(std::memory_order_relaxed);
            }
            result.flux_in = cell->flux_in.load(std::memory_order_relaxed);
            result.flux_out = cell->flux_out.load(std::memory_order_relaxed);
        }

        return result;
    }

    void advectStep() override {
        // Esquema de advección upwind simplificado
        for (auto& [cell_id, cell] : local_data) {
            int left_cell = cell_id - 1;

            auto it_left = local_data.find(left_cell);
            if (it_left != local_data.end()) {
                auto left_data = getCell(left_cell);

                for (int i = 0; i < num_species; ++i) {
                    double current = cell->concentrations[i].load(std::memory_order_relaxed);
                    double updated = current + (left_data.concentrations[i] - current);
                    cell->concentrations[i].store(updated, std::memory_order_relaxed);
                }
            }
        }
    }

    void syncGhostCells() override {
        // Sincronización simple entre procesos
        MPI_Barrier(MPI_COMM_WORLD);
        // (Futuro: implementar comunicación de bordes usando MPI_Sendrecv)
    }

    std::string getStrategyName() const override {
        return "Lock-Free";
    }
};

#endif // LOCK_FREE_HASH_TABLE_HPP

#include <chrono>
#include <iostream>
#include <memory>
#include <utility>
#include <mpi.h>

// Cabeceras del proyecto
#include "distributed_hash_table.hpp"
#include "lock_free_hash_table.hpp"
#include "coarse_grained_hash_table.hpp"
#include "fine_grained_hash_table.hpp"

class POETSimulator {
private:
    std::unique_ptr<DistributedHashTable> hash_table;
    SimulationParams params;
    int rank, size;
    
public:
    POETSimulator(std::unique_ptr<DistributedHashTable>&& table, 
                  const SimulationParams& params, int rank, int size)
        : hash_table(std::move(table)), params(params), rank(rank), size(size) {}
    
    void runSimulation() {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int step = 0; step < params.steps; ++step) {
            if (rank == 0 && step % 100 == 0) {
                std::cout << "Step " << step << std::endl;
            }
            
            hash_table->advectStep();
            hash_table->syncGhostCells();
            
            // Simular reacciones químicas simples
            simulateReactions();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
            
        if (rank == 0) {
            std::cout << hash_table->getStrategyName() 
                      << " simulation completed in " << duration.count() 
                      << " ms" << std::endl;
        }
    }
    
private:
    void simulateReactions() {
        // Reacciones químicas simplificadas
        // Ejemplo: A + B -> C
        for (int i = 0; i < params.grid_x * params.grid_y / size; ++i) {
            int cell_id = rank * (params.grid_x * params.grid_y / size) + i;
            auto cell = hash_table->getCell(cell_id);
            
            if (cell.concentrations.size() >= 3) {
                double reaction_rate = 0.01;
                double delta = cell.concentrations[0] * cell.concentrations[1] * 
                              reaction_rate * params.dt;
                
                cell.concentrations[0] -= delta;
                cell.concentrations[1] -= delta;
                cell.concentrations[2] += delta;
                
                hash_table->updateCell(cell_id, cell);
            }
        }
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    SimulationParams params;
    params.grid_x = 500;
    params.grid_y = 1500;
    params.num_species = 5;
    params.steps = 200;
    
    int total_cells = params.grid_x * params.grid_y;
    
    // Probar diferentes estrategias
    if (rank == 0) {
        std::cout << "=== POET Simplified Benchmark ===" << std::endl;
    }
    
    // Lock-Free
    if (rank == 0) std::cout << "\nTesting Lock-Free..." << std::endl;
    auto lock_free_table = std::make_unique<LockFreeHashTable>(
        params.num_species, total_cells, rank, size);
    POETSimulator lock_free_sim(std::move(lock_free_table), params, rank, size);
    lock_free_sim.runSimulation();
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Coarse-Grained
    if (rank == 0) std::cout << "\nTesting Coarse-Grained Locking..." << std::endl;
    auto coarse_table = std::make_unique<CoarseGrainedHashTable>(
        params.num_species, total_cells, rank, size);
    POETSimulator coarse_sim(std::move(coarse_table), params, rank, size);
    coarse_sim.runSimulation();
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Fine-Grained
    if (rank == 0) std::cout << "\nTesting Fine-Grained Locking..." << std::endl;
    auto fine_table = std::make_unique<FineGrainedHashTable>(
        params.num_species, total_cells, rank, size);
    POETSimulator fine_sim(std::move(fine_table), params, rank, size);
    fine_sim.runSimulation();
    
    MPI_Finalize();
    return 0;
}
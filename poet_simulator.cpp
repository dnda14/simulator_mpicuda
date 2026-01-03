#include <chrono>
#include <iostream>
#include <memory>
#include <utility>
#include <mpi.h>
#include <vector>

// Cabeceras del proyecto
// Asegúrate de que estos archivos tengan el código NUEVO que generamos
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
        // Inicializar celdas con valores de concentración
        initializeCells();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int step = 0; step < params.steps; ++step) {
            if (rank == 0 && step % 50 == 0) { // Imprimir cada 50 pasos para ver progreso
                std::cout << "Step " << step << " running..." << std::endl;
            }
            
            // 1. Advección
            // NOTA: En la implementación "Lite" del benchmark, nos saltamos la física de fluidos
            // para centrarnos en el estrés de lectura/escritura de la DHT.
            // hash_table->advectStep(); 
            
            // 2. Sincronización
            hash_table->syncGhostCells();
            
            // 3. Reacciones (Aquí ocurre la carga pesada sobre la DHT)
            simulateReactions();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
            
        if (rank == 0) {
            std::cout << ">>> RESULT: " << hash_table->getStrategyName() 
                      << " completed in " << duration.count() 
                      << " ms" << std::endl;
        }
    }
    
private:
    // Inicializar concentraciones con valores no-cero
    void initializeCells() {
        int cells_per_rank = (params.grid_x * params.grid_y) / size;
        int start_id = rank * cells_per_rank;
        int end_id = start_id + cells_per_rank;

        for (int cell_id = start_id; cell_id < end_id; ++cell_id) {
            GridCell cell;
            // Inicializar con gradiente para simular condiciones iniciales
            double x = (cell_id % params.grid_x) / (double)params.grid_x;
            double y = (cell_id / params.grid_x) / (double)params.grid_y;
            
            cell.concentrations[0] = 1.0 - x;        // Especie A: gradiente horizontal
            cell.concentrations[1] = y;              // Especie B: gradiente vertical
            cell.concentrations[2] = 0.0;            // Especie C: producto
            cell.concentrations[3] = 0.5;            // Especie D: constante
            cell.concentrations[4] = (x + y) / 2.0;  // Especie E: mixto
            
            hash_table->updateCell(cell_id, cell);
        }
        hash_table->syncGhostCells();
    }

    void simulateReactions() {
        int cells_per_rank = (params.grid_x * params.grid_y) / size;
        int start_id = rank * cells_per_rank;
        int end_id = start_id + cells_per_rank;

        double diffusion_coef = 0.1;
        double reaction_rate = 0.01;

        for (int cell_id = start_id; cell_id < end_id; ++cell_id) {
            // Coordenadas 2D de la celda
            int x = cell_id % params.grid_x;
            int y = cell_id / params.grid_x;
            
            // A. READ celda actual
            auto cell = hash_table->getCell(cell_id);
            
            // B. READ celdas vecinas (acceso potencialmente remoto - aumenta contención)
            GridCell left, right, up, down;
            
            // Condiciones de borde periódicas
            int left_id  = (x > 0) ? cell_id - 1 : cell_id + params.grid_x - 1;
            int right_id = (x < params.grid_x - 1) ? cell_id + 1 : cell_id - params.grid_x + 1;
            int up_id    = (y > 0) ? cell_id - params.grid_x : cell_id + (params.grid_y - 1) * params.grid_x;
            int down_id  = (y < params.grid_y - 1) ? cell_id + params.grid_x : cell_id - (params.grid_y - 1) * params.grid_x;
            
            left  = hash_table->getCell(left_id);
            right = hash_table->getCell(right_id);
            up    = hash_table->getCell(up_id);
            down  = hash_table->getCell(down_id);
            
            // C. DIFUSIÓN (Laplaciano discreto)
            for (int s = 0; s < params.num_species; ++s) {
                double laplacian = left.concentrations[s] + right.concentrations[s] 
                                 + up.concentrations[s] + down.concentrations[s] 
                                 - 4.0 * cell.concentrations[s];
                cell.concentrations[s] += diffusion_coef * laplacian * params.dt;
            }
            
            // D. REACCIÓN QUÍMICA (A + B -> C)
            double delta = cell.concentrations[0] * cell.concentrations[1] * reaction_rate * params.dt;
            cell.concentrations[0] -= delta;
            cell.concentrations[1] -= delta;
            cell.concentrations[2] += delta;
            
            // E. WRITE resultado
            hash_table->updateCell(cell_id, cell);
        }
    }
};

int main(int argc, char** argv) {
    // Inicialización MPI estándar
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    SimulationParams params;
    params.grid_x = 500;
    params.grid_y = 1500;
    params.num_species = 5;
    params.steps = 50; // Reducido para pruebas rápidas (puedes subirlo a 200 luego)
    
    int total_cells = params.grid_x * params.grid_y;
    
    if (rank == 0) {
        std::cout << "==========================================" << std::endl;
        std::cout << "   POET DISTRIBUTED BENCHMARK (MPI RMA)   " << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Grid: " << params.grid_x << "x" << params.grid_y 
                  << " | Processes: " << size << std::endl;
    }

    // ---------------------------------------------------------
    // 1. Test Lock-Free (Optimistic Checksum)
    // ---------------------------------------------------------
    if (rank == 0) std::cout << "\n[1/3] Testing Lock-Free Strategy..." << std::endl;
    
    // IMPORTANTE: MPI_Barrier para asegurar que todos inicien juntos
    MPI_Barrier(MPI_COMM_WORLD); 
    
    { // Scope para destruir el objeto antes de pasar al siguiente
        auto lock_free_table = std::make_unique<LockFreeHashTable>(
            total_cells, rank, size); // Constructor corregido
        POETSimulator lock_free_sim(std::move(lock_free_table), params, rank, size);
        lock_free_sim.runSimulation();
    }
    
    // ---------------------------------------------------------
    // 2. Test Coarse-Grained (Global Window Lock)
    // ---------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "\n[2/3] Testing Coarse-Grained Locking..." << std::endl;
    
    {
        auto coarse_table = std::make_unique<CoarseGrainedHashTable>(
            total_cells, rank, size); // Constructor corregido
        POETSimulator coarse_sim(std::move(coarse_table), params, rank, size);
        coarse_sim.runSimulation();
    }
    
    // ---------------------------------------------------------
    // 3. Test Fine-Grained (CAS - Atomic Operations)
    // ---------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "\n[3/3] Testing Fine-Grained Locking..." << std::endl;

    {
        auto fine_table = std::make_unique<FineGrainedHashTable>(
            total_cells, rank, size); // Constructor corregido
        POETSimulator fine_sim(std::move(fine_table), params, rank, size);
        fine_sim.runSimulation();
    }
    
    if (rank == 0) std::cout << "\nAll benchmarks finished." << std::endl;
    
    MPI_Finalize();
    return 0;
}
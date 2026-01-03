#ifndef COARSE_GRAINED_HASH_TABLE_HPP
#define COARSE_GRAINED_HASH_TABLE_HPP

#include "distributed_hash_table.hpp"

class CoarseGrainedHashTable : public DistributedHashTable {
public:
    CoarseGrainedHashTable(int total_entries, int rank, int size)
        : DistributedHashTable(total_entries, rank, size) {}

    // Escritura Remota (Sección 3.1 del Paper)
    void updateCell(int key, const GridCell& val) override {
        int target_rank = getOwnerRank(key);
        // Desplazamiento inicial (hash)
        MPI_Aint target_offset = getLocalOffset(key);
        
        // 1. BLOQUEO GRUESO (The Bottleneck)
        // "Whenever an DHT_read or DHT_write operation is initiated... 
        // the entire memory window... is locked." [cite: 146-147]
        // Usamos LOCK_EXCLUSIVE para escrituras.
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_rank, 0, win);

        DHT_Bucket temp;
        bool written = false;
        int attempts = 0;
        const int MAX_ATTEMPTS = 50; // Evitar loop infinito si está lleno

        // 2. COLLISION HANDLING (Linear Probing)
        // "If the bucket is already occupied... the bucket at the next index is checked" [cite: 140]
        while (attempts < MAX_ATTEMPTS) {
            // Leemos el bucket remoto para ver su estado
            MPI_Get(&temp, sizeof(DHT_Bucket), MPI_BYTE,
                    target_rank, target_offset * sizeof(DHT_Bucket),
                    sizeof(DHT_Bucket), MPI_BYTE, win);
            
            // Forzamos que la lectura termine antes de verificar (Flush local)
            MPI_Win_flush(target_rank, win);

            // Verificamos si podemos escribir aquí
            if (temp.status == 0 || temp.key == key) {
                // Preparamos el bucket a escribir
                temp.key = key;
                temp.value = val;
                temp.status = 1; // Ocupado

                // Escribimos (Remote Write)
                MPI_Put(&temp, sizeof(DHT_Bucket), MPI_BYTE,
                        target_rank, target_offset * sizeof(DHT_Bucket),
                        sizeof(DHT_Bucket), MPI_BYTE, win);
                
                written = true;
                break;
            }

            // Colisión: Intentar siguiente slot
            target_offset = (target_offset + 1) % local_capacity;
            attempts++;
        }

        // 3. DESBLOQUEO
        // "The lock is released with MPI_Win_unlock" [cite: 149]
        MPI_Win_unlock(target_rank, win);
    }

    // Lectura Remota
    GridCell getCell(int key) override {
        int target_rank = getOwnerRank(key);
        MPI_Aint target_offset = getLocalOffset(key);
        GridCell result; // Por defecto vacía

        // Usamos LOCK_SHARED para lecturas (permite múltiples lectores) [cite: 148]
        MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, win);

        DHT_Bucket temp;
        int attempts = 0;
        const int MAX_ATTEMPTS = 50;

        while (attempts < MAX_ATTEMPTS) {
            // Leer bucket remoto
            MPI_Get(&temp, sizeof(DHT_Bucket), MPI_BYTE,
                    target_rank, target_offset * sizeof(DHT_Bucket),
                    sizeof(DHT_Bucket), MPI_BYTE, win);
            
            MPI_Win_flush(target_rank, win);

            if (temp.status == 0) {
                // Llegamos a un hueco vacío -> La clave no existe
                break;
            }
            
            if (temp.key == key) {
                // ¡Encontrado!
                result = temp.value;
                break;
            }

            // Seguir buscando (Linear Probing)
            target_offset = (target_offset + 1) % local_capacity;
            attempts++;
        }

        MPI_Win_unlock(target_rank, win);
        return result;
    }

    std::string getStrategyName() const override {
        return "Coarse-Grained (MPI_Win_lock)";
    }

    // Override: No usamos lock_all, así que solo sincronizamos con Barrier
    void syncGhostCells() override {
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

#endif // COARSE_GRAINED_HASH_TABLE_HPP
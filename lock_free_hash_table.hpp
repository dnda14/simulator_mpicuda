#ifndef LOCK_FREE_HASH_TABLE_HPP
#define LOCK_FREE_HASH_TABLE_HPP

#include "distributed_hash_table.hpp"

class LockFreeHashTable : public DistributedHashTable {
public:
    LockFreeHashTable(int total_entries, int rank, int size)
        : DistributedHashTable(total_entries, rank, size) {
        
        // ESTRATEGIA: RMA Pasivo Continuo
        // "All windows are locked by all processes with MPI_Win_lock_all" 
        // Esto elimina el overhead de adquirir/liberar locks en cada operación.
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    }

    ~LockFreeHashTable() {
        MPI_Win_unlock_all(win);
    }

    // Función auxiliar de Checksum (Hash simple para integridad)
    // "The origin process is responsible for calculating a checksum" [cite: 245]
    unsigned int calculateChecksum(const DHT_Bucket& b) {
        unsigned int hash = 0;
        // Checksum de la clave y los datos (concentraciones)
        hash ^= std::hash<int>{}(b.key);
        for(double c : b.value.concentrations) {
            hash ^= std::hash<double>{}(c) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }

    void updateCell(int key, const GridCell& val) override {
        int target_rank = getOwnerRank(key);
        MPI_Aint target_offset = getLocalOffset(key);

        DHT_Bucket bucket;
        bucket.key = key;
        bucket.value = val;
        bucket.status = 1; // Ocupado
        
        // 1. CALCULAR CHECKSUM
        // "Appending it to the bucket data" [cite: 245]
        bucket.checksum = calculateChecksum(bucket);

        // 2. ESCRITURA "OPTIMISTA" (Sin Lock individual)
        // Usamos MPI_Put directamente. Si hay colisión de escritura, el checksum del lector fallará.
        MPI_Put(&bucket, sizeof(DHT_Bucket), MPI_BYTE,
                target_rank, target_offset * sizeof(DHT_Bucket),
                sizeof(DHT_Bucket), MPI_BYTE, win);
        
        // Asegurar que el dato salga del buffer local hacia la red
        MPI_Win_flush(target_rank, win);
    }

    GridCell getCell(int key) override {
        int target_rank = getOwnerRank(key);
        MPI_Aint target_offset = getLocalOffset(key);
        
        DHT_Bucket temp;
        int attempts = 0;
        const int MAX_ATTEMPTS = 10; // Límite de reintentos por consistencia

        while (attempts < MAX_ATTEMPTS) {
            // 1. LEER (READ)
            MPI_Get(&temp, sizeof(DHT_Bucket), MPI_BYTE,
                    target_rank, target_offset * sizeof(DHT_Bucket),
                    sizeof(DHT_Bucket), MPI_BYTE, win);
            
            MPI_Win_flush(target_rank, win); // Esperar a recibir datos

            // Si está vacío, no hay nada que validar
            if (temp.status == 0) return GridCell();

            // 2. VALIDAR CHECKSUM
            // "Recalculates the checksum... If equal... returned" [cite: 246-247]
            unsigned int local_calc = calculateChecksum(temp);
            
            if (local_calc == temp.checksum) {
                if (temp.key == key) return temp.value;
                // Colisión de Hash (Linear Probing) - no implementado full en versión simple
                // para mantener el benchmark enfocado en la latencia de red/consistencia.
                return GridCell(); 
            }

            // 3. FALLO DE CONSISTENCIA -> REINTENTAR
            // "In the event of a mismatch, the MPI_Get operation... is repeated" [cite: 248]
            attempts++;
            // (Opcional) Pequeño backoff para dejar que el escritor termine
        }
        
        // Si falla muchas veces, devolvemos celda vacía o error (simulado)
        return GridCell(); 
    }

    std::string getStrategyName() const override {
        return "Lock-Free (Optimistic Checksum)";
    }
};

#endif // LOCK_FREE_HASH_TABLE_HPP
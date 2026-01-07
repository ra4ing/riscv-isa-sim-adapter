// Copyright (c) 2024-2025 DiveFuzz Project
// SPDX-License-Identifier: Mulan PSL v2

#ifndef SPIKE_ENGINE_H
#define SPIKE_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <map>

// Forward declarations
class sim_t;
class processor_t;
class cfg_t;

namespace spike_engine {

/**
 * Checkpoint state for processor
 *
 * Represents a lightweight snapshot of processor state at a specific point.
 * Designed for efficient rollback during instruction validation (e.g., when
 * XOR value collisions occur and we need to retry with different instructions).
 *
 * Design Philosophy:
 *   - Minimal footprint: Only stores essential state (registers, PC, index)
 *   - Fast restore: Uses vector truncation instead of memory writes
 *   - Single checkpoint: Optimized for the common pattern of one active checkpoint
 *
 * Memory Consistency:
 *   After restore, memory may contain instructions that were executed after
 *   the checkpoint, but this is safe because:
 *   1. PC and next_instruction_addr point to the checkpoint position
 *   2. Subsequent execute_sequence() calls will overwrite old instructions
 *   3. Instructions past the checkpoint position will never be executed
 */
struct Checkpoint {
    // General purpose registers (x0-x31)
    std::vector<uint64_t> xpr;

    // Floating point registers (f0-f31)
    // freg_t has two 64-bit fields: v[0] and v[1]
    // Both must be saved/restored for correct NaN-boxing behavior
    std::vector<uint64_t> fpr;      // FPR[i].v[0] - main 64-bit value
    std::vector<uint64_t> fpr_v1;   // FPR[i].v[1] - extended/internal state

    // Program counter
    uint64_t pc;

    // Index into instruction sequence
    // Marks how many instructions have been executed at checkpoint time
    size_t instr_index;

    // Next available memory address for instruction placement
    uint64_t next_instruction_addr;

    // Memory region backup for checkpoint/restore
    // This is essential for correct rollback when instructions modify memory
    // (e.g., AMO instructions, store instructions)
    std::vector<uint8_t> mem_region_backup;

    // ========== Privilege and Mode State ==========
    // Privilege level (0=U, 1=S, 3=M)
    // Essential for correct trap handling after restore
    uint64_t prv;

    // Virtualization mode (for H extension)
    bool v;

    // Debug mode flag
    bool debug_mode;

    // ========== All CSRs ==========
    // Complete CSR state captured as address->value map
    // This ensures ALL CSRs are properly restored, including:
    // - Trap handling CSRs (mtvec, mepc, mcause, mtval, etc.)
    // - Status registers (mstatus, sstatus, etc.)
    // - Interrupt/exception delegation registers
    // - PMP configuration
    // - Floating-point CSRs (fflags, frm, fcsr)
    // - Custom CSRs
    std::map<uint64_t, uint64_t> csr_values;

    Checkpoint();
};

/**
 * SpikeEngine: Efficient Spike execution engine with checkpointing
 *
 * A lightweight wrapper around the Spike RISC-V ISA simulator, designed for
 * instruction-by-instruction execution with support for mixed-length instructions
 * (16-bit compressed and 32-bit standard) and efficient checkpoint/restore.
 *
 * Key Features:
 *   - Unified execution model: One method handles all cases (single, jump, loop)
 *   - Mixed-length instruction support (RV*C compressed + standard)
 *   - Dynamic address allocation (no pre-allocated slots)
 *   - Lightweight checkpointing (registers + index only)
 *   - Python layer handles XOR validation (simpler, more flexible)
 *
 * Typical Workflow:
 *   1. Initialize with pre-compiled ELF template (containing nop placeholder region)
 *   2. Engine executes template initialization code until reaching main()
 *   3. For each instruction to test:
 *      a. set_checkpoint() - save current processor state
 *      b. Read source register values (Python: get_xpr/get_fpr)
 *      c. execute_sequence() - write and execute instruction(s)
 *      d. Read destination register values (Python: get_xpr/get_fpr)
 *      e. Python computes XOR and checks uniqueness
 *      f. If collision: restore_checkpoint() and retry
 *         If unique: proceed to next instruction
 *
 * Memory Management:
 *   Instructions are written sequentially starting from main symbol address.
 *   2-byte compressed instructions are detected automatically and written with
 *   correct size. PC tracking ensures proper alignment for mixed-length sequences.
 *
 * Thread Safety:
 *   Not thread-safe. Each thread should use its own SpikeEngine instance.
 *
 * Example:
 *   SpikeEngine engine("template.elf", "rv64gc", 1000);
 *   engine.initialize();
 *
 *   engine.set_checkpoint();
 *   uint64_t src_val = engine.get_xpr(2);  // Read x2 before
 *   engine.execute_sequence({0x003100b3}, {4});  // add x1, x2, x3
 *   uint64_t dst_val = engine.get_xpr(1);  // Read x1 after
 *   // Python: compute XOR, check uniqueness, etc.
 */
class SpikeEngine {
public:
    /**
     * Constructor
     * @param elf_path Path to pre-compiled ELF file
     * @param isa ISA string (e.g., "rv64gc")
     * @param num_instrs Number of instructions to generate (nops in ELF)
     * @param verbose Enable verbose output (default: false)
     */
    SpikeEngine(const std::string& elf_path,
                const std::string& isa,
                size_t num_instrs,
                bool verbose = false);

    /**
     * Destructor
     */
    ~SpikeEngine();

    /**
     * Initialize Spike and run until main function entry
     * This executes template initialization code
     * @return true on success, false on error
     */
    bool initialize();

    /**
     * Create a checkpoint of current processor state
     * Saves PC, registers, and modified memory
     */
    void set_checkpoint();

    /**
     * Restore processor state from last checkpoint
     */
    void restore_checkpoint();

    /**
     * Execute a sequence of instructions
     *
     * Unified execution method that handles all cases:
     * - Single instruction: execute_sequence([code], [size])
     * - Forward jump: execute_sequence([jump, middle...], [sizes...])
     * - Backward loop: execute_sequence([init, body..., decr, branch], [sizes...])
     *
     * Execution logic:
     * 1. Write all instructions to memory starting at next_instruction_addr_
     * 2. Calculate target_pc = next_instruction_addr_ + sum(sizes)
     * 3. Execute until PC == target_pc
     * 4. Update next_instruction_addr_ to target_pc
     *
     * All cases are handled uniformly:
     * - Sequential: PC advances to target_pc after each step
     * - Trap: Spike handles internally, mret returns to next instruction
     * - Forward jump: PC jumps directly to target
     * - Backward loop: PC loops until branch falls through
     *
     * @param machine_codes List of machine codes to execute
     * @param sizes List of instruction sizes (2 or 4 bytes each)
     * @param max_steps Maximum execution steps (safety limit, default: 10000)
     * @return Number of instructions in the sequence
     *
     * @throws std::runtime_error if:
     *   - Engine not initialized
     *   - Out of instruction region space
     *   - PC mismatch (internal consistency error)
     *   - Memory write failure
     *   - max_steps exceeded before reaching target_pc
     */
    size_t execute_sequence(
        const std::vector<uint32_t>& machine_codes,
        const std::vector<size_t>& sizes,
        size_t max_steps = 10000);

    /**
     * Get value of a general-purpose register
     * @param reg_index Register index (0-31)
     * @return Register value
     */
    uint64_t get_xpr(int reg_index) const;

    /**
     * Get value of a floating-point register
     * @param reg_index Register index (0-31)
     * @return Register value (as uint64_t)
     */
    uint64_t get_fpr(int reg_index) const;

    /**
     * Get program counter value
     * @return PC value
     */
    uint64_t get_pc() const;

    /**
     * Get all general-purpose register values
     * @return Vector of 32 register values (x0-x31)
     */
    std::vector<uint64_t> get_all_xpr() const;

    /**
     * Get all floating-point register values
     * @return Vector of 32 register values (f0-f31)
     */
    std::vector<uint64_t> get_all_fpr() const;

    /**
     * Get a CSR value by address
     * @param csr_addr CSR address (e.g., 0x300 for mstatus)
     * @return CSR value, or 0 if not found/accessible
     */
    uint64_t get_csr(uint64_t csr_addr) const;

    /**
     * Get all accessible CSR values
     * @return Map of CSR address -> value
     */
    std::map<uint64_t, uint64_t> get_all_csrs() const;

    /**
     * Get mem_region start address
     * @return Start address of mem_region (for testing memory operations)
     */
    uint64_t get_mem_region_start() const { return mem_region_start_; }

    /**
     * Get mem_region size
     * @return Size of mem_region in bytes
     */
    size_t get_mem_region_size() const { return mem_region_size_; }

    /**
     * Read memory at specified address
     * @param addr Memory address to read from
     * @param size Number of bytes to read
     * @return Vector of bytes read from memory
     */
    std::vector<uint8_t> read_mem(uint64_t addr, size_t size) const;

    /**
     * Get current instruction index
     * @return Index of next instruction to replace
     */
    size_t get_current_index() const { return current_instr_index_; }

    /**
     * Get total number of instructions
     */
    size_t get_num_instrs() const { return num_instrs_; }

    /**
     * Get last error message
     */
    std::string get_last_error() const { return last_error_; }

    /**
     * Check if the last executed instruction triggered a trap/exception.
     * This is useful for logging - instructions that cause traps are handled
     * by the exception handler (which skips them), but they are still "accepted"
     * from the fuzzer's perspective.
     *
     * @return true if the last instruction triggered a trap, false otherwise
     */
    bool was_last_execution_trapped() const { return last_execution_trapped_; }

    /**
     * Get the number of trap handler steps executed in the last execution.
     * Returns 0 if no trap occurred.
     *
     * @return Number of steps executed in trap handler
     */
    size_t get_last_trap_handler_steps() const { return last_trap_handler_steps_; }

    /**
     * Detect instruction size from machine code (static utility)
     *
     * Determines whether an instruction is 16-bit compressed or 32-bit standard
     * based on RISC-V encoding rules specified in the ISA manual.
     *
     * RISC-V Instruction Length Encoding (bits[1:0]):
     *   - bits[1:0] != 0b11  →  16-bit compressed instruction (RV*C)
     *   - bits[1:0] == 0b11  →  32-bit or longer instruction
     *     - If bits[4:2] != 0b111  →  32-bit standard instruction
     *     - If bits[4:2] == 0b111  →  48/64-bit instruction (not supported)
     *
     * @param machine_code 32-bit instruction encoding (may contain zero-padded
     *                     16-bit compressed instruction in lower half)
     *
     * @return Instruction size in bytes: 2 for compressed, 4 for standard
     *
     * @throws std::runtime_error if instruction length > 32 bits (unsupported)
     *
     * @note This is a static method and can be called without an instance:
     *       size_t sz = SpikeEngine::get_instruction_size(0x00008522);
     */
    static size_t get_instruction_size(uint32_t machine_code);

private:
    // Configuration
    std::string elf_path_;
    std::string isa_;
    size_t num_instrs_;
    bool verbose_;

    // Spike simulator
    std::unique_ptr<cfg_t> cfg_;
    std::unique_ptr<sim_t> sim_;
    processor_t* proc_;  // Main processor (core 0)

    // Instruction region
    uint64_t instruction_region_start_;
    uint64_t instruction_region_end_;
    uint64_t next_instruction_addr_;  // Next available address for instruction

    // Memory region for checkpoint/restore (e.g., .mem_region section)
    uint64_t mem_region_start_;
    size_t mem_region_size_;

    // Execution state
    size_t current_instr_index_;

    Checkpoint checkpoint_;
    bool checkpoint_valid_;
    bool initialized_;

    // Error handling
    std::string last_error_;

    // Trap detection (for logging)
    bool last_execution_trapped_;      // True if last instruction triggered a trap
    size_t last_trap_handler_steps_;   // Steps executed in trap handler (0 if no trap)

    // Internal helper methods

    /**
     * Find instruction region using ELF symbols
     * Reads 'main' symbol from ELF to locate instruction region
     */
    bool find_nop_region();

    /**
     * Read symbol address from ELF file
     * Uses objdump to extract symbol address
     * @param symbol_name Symbol name to look up (e.g., "main")
     * @return Symbol address, or 0 if not found
     */
    uint64_t read_symbol_address(const std::string& symbol_name);

    /**
     * Write machine code to memory address
     * @param addr Physical address
     * @param code Machine code
     * @param size Instruction size (2 or 4 bytes)
     */
    bool write_memory(uint64_t addr, uint32_t code, size_t size);

    /**
     * Read memory from physical address
     * @param addr Physical address
     * @return 32-bit value
     */
    uint32_t read_memory(uint64_t addr);

    /**
     * Single step execution (used during initialization)
     * Executes one instruction on processor
     */
    bool step_processor();

    /**
     * Save processor state to checkpoint
     */
    void save_state(Checkpoint& checkpoint);

    /**
     * Restore processor state from checkpoint
     */
    void restore_state(const Checkpoint& checkpoint);
};

} // namespace spike_engine

#endif // SPIKE_ENGINE_H

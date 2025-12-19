#include "spike_engine.h"
#include "../riscv/sim.h"
#include "../riscv/processor.h"
#include "../riscv/mmu.h"
#include "../riscv/cfg.h"
#include "../riscv/decode.h"
#include "../riscv/trap.h"

#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <algorithm>

namespace spike_engine {

//==============================================================================
// Checkpoint Implementation
//==============================================================================

Checkpoint::Checkpoint()
    : xpr(32, 0)
    , fpr(32, 0)
    , pc(0)
    , instr_index(0)
{
}

//==============================================================================
// SpikeEngine Implementation
//==============================================================================

SpikeEngine::SpikeEngine(const std::string& elf_path,
                         const std::string& isa,
                         size_t num_instrs,
                         bool verbose)
    : elf_path_(elf_path)
    , isa_(isa)
    , num_instrs_(num_instrs)
    , verbose_(verbose)
    , proc_(nullptr)
    , instruction_region_start_(0)
    , instruction_region_end_(0)
    , next_instruction_addr_(0)
    , current_instr_index_(0)
    , checkpoint_valid_(false)
    , initialized_(false)
{
}

size_t SpikeEngine::get_instruction_size(uint32_t machine_code) {
    // RISC-V instruction length encoding (from RISC-V spec):
    // - If bits[1:0] != 0b11, it's a 16-bit compressed instruction
    // - If bits[1:0] == 0b11 and bits[4:2] != 0b111, it's a 32-bit instruction
    // - Otherwise, it's a longer instruction (48/64-bit, not commonly used)

    uint8_t opcode_low = machine_code & 0x3;

    if (opcode_low != 0x3) {
        // Compressed instruction (16-bit)
        return 2;
    }

    uint8_t opcode_mid = (machine_code >> 2) & 0x7;
    if (opcode_mid != 0x7) {
        // Standard 32-bit instruction
        return 4;
    }

    // Instructions longer than 32 bits are not supported
    throw std::runtime_error("Unsupported instruction length (>32 bits)");
}

SpikeEngine::~SpikeEngine() {
    // sim_ will be automatically destroyed
}

bool SpikeEngine::initialize() {
    try {
        // Create configuration
        cfg_.reset(new cfg_t());
        cfg_->isa = isa_.c_str();
        cfg_->priv = "MSU";
        cfg_->misaligned = false;
        cfg_->endianness = endianness_little;
        cfg_->pmpregions = 16;
        cfg_->hartids = std::vector<size_t>{0};
        cfg_->real_time_clint = false;
        cfg_->trigger_count = 4;

        // Set memory layout
        cfg_->mem_layout.push_back(mem_cfg_t(0x80000000, 0x10000000));

        // Setup memory regions
        // We need two memory regions:
        // 1. Low memory for HTIF boot code (avoid [0, 0x1000) which Spike reserves)
        // 2. Main memory for our ELF (0x80000000 - 0x90000000)
        std::vector<std::pair<reg_t, abstract_mem_t*>> mems;

        // Low memory for boot code (starting after Spike's reserved region)
        mems.push_back(std::make_pair(0x1000, new mem_t(0xF000)));  // 60KB from 0x1000
        if (verbose_) {
            std::cout << "[SpikeEngine] Created boot memory: 0x1000 - 0x10000" << std::endl;
        }

        // Main memory for ELF
        reg_t mem_base = 0x80000000;
        size_t mem_size = 0x10000000; // 256MB
        mems.push_back(std::make_pair(mem_base, new mem_t(mem_size)));
        if (verbose_) {
            std::cout << "[SpikeEngine] Created main memory: 0x" << std::hex << mem_base
                      << " - 0x" << (mem_base + mem_size) << std::dec << std::endl;
        }

        // Create simulator
        std::vector<std::string> htif_args = {elf_path_};
        std::vector<device_factory_sargs_t> plugin_devices;
        debug_module_config_t dm_config;

        sim_.reset(new sim_t(
            cfg_.get(),
            /*halted=*/false,
            mems,  // Empty - let HTIF/ELF determine memory layout
            plugin_devices,
            htif_args,
            dm_config,
            /*log_path=*/nullptr,
            /*dtb_enabled=*/false,  // Disable DTB for simple testing
            /*dtb_file=*/nullptr,
            /*socket_enabled=*/false,
            /*cmd_file=*/nullptr,
            /*instruction_limit=*/std::nullopt
        ));

        // CRITICAL: Start HTIF to load ELF program
        // htif_t::start() will:
        // 1. Call load_program() to parse ELF and load into memory
        // 2. Call reset() to initialize processor
        if (verbose_) {
            std::cout << "[SpikeEngine] Starting HTIF to load ELF..." << std::endl;
        }
        sim_->start();
        if (verbose_) {
            std::cout << "[SpikeEngine] HTIF start() completed" << std::endl;
        }

        // Get processor 0 (after HTIF start)
        proc_ = sim_->get_core(0);
        if (!proc_) {
            last_error_ = "Failed to get processor 0";
            return false;
        }

        // Check PC after HTIF initialization
        uint64_t boot_pc = proc_->get_state()->pc;
        if (verbose_) {
            std::cout << "[SpikeEngine] PC after HTIF start (boot address): 0x" << std::hex << boot_pc << std::dec << std::endl;
        }

        // Check if boot address has valid instructions (HTIF boot code)
        if (verbose_) {
            std::cout << "[SpikeEngine] Reading memory at boot address..." << std::endl;
        }
        uint32_t boot_instr = 0;
        try {
            boot_instr = read_memory(boot_pc);
            if (verbose_) {
                std::cout << "[SpikeEngine] Instruction at boot address: 0x" << std::hex << boot_instr << std::dec << std::endl;
            }
        } catch (...) {
            if (verbose_) {
                std::cout << "[SpikeEngine] Failed to read memory at boot address" << std::endl;
            }
            throw;
        }

        // Find main symbol address
        if (!find_nop_region()) {
            last_error_ = "Failed to find main symbol in ELF";
            return false;
        }

        // Find _start symbol
        uint64_t start_addr = read_symbol_address("_start");
        if (verbose_) {
            std::cout << "[SpikeEngine] _start address: 0x" << std::hex << start_addr << std::dec << std::endl;
        }

        // Verify _start has valid instructions
        uint32_t start_instr = read_memory(start_addr);
        if (verbose_) {
            std::cout << "[SpikeEngine] Instruction at _start: 0x" << std::hex << start_instr << std::dec << std::endl;
        }

        if (start_instr == 0) {
            last_error_ = "Memory at _start is empty - ELF not loaded properly";
            return false;
        }

        // If boot PC has valid code, execute from boot to _start
        // Otherwise, directly set PC to _start
        uint64_t entry_pc;
        if (boot_instr != 0 && boot_pc != start_addr) {
            if (verbose_) {
                std::cout << "[SpikeEngine] Boot code found, executing from boot to _start..." << std::endl;
            }
            entry_pc = boot_pc;
        } else {
            if (verbose_) {
                std::cout << "[SpikeEngine] No boot code, setting PC directly to _start" << std::endl;
            }
            proc_->get_state()->pc = start_addr;
            entry_pc = start_addr;
        }

        if (verbose_) {
            std::cout << "[SpikeEngine] main address: 0x" << std::hex << instruction_region_start_ << std::dec << std::endl;
        }

        // Execute from entry to main
        if (entry_pc != instruction_region_start_) {
            const size_t max_steps = 100000;
            size_t steps = 0;

            if (verbose_) {
                std::cout << "[SpikeEngine] Executing from entry (0x" << std::hex << entry_pc
                          << ") to main (0x" << instruction_region_start_ << ")" << std::dec << std::endl;
            }

            while (proc_->get_state()->pc != instruction_region_start_ && steps < max_steps) {
                uint64_t current_pc = proc_->get_state()->pc;

                if (!step_processor()) {
                    std::ostringstream oss;
                    oss << "Failed at PC 0x" << std::hex << current_pc;
                    last_error_ = oss.str();
                    return false;
                }

                steps++;

                // Debug: Print PC every 10000 steps
                if (verbose_ && steps % 10000 == 0) {
                    std::cout << "[DEBUG] Step " << steps << ", PC: 0x"
                              << std::hex << proc_->get_state()->pc << std::dec << std::endl;
                }
            }

            if (steps >= max_steps) {
                std::ostringstream oss;
                oss << "Initialization timeout: PC did not reach main"
                    << " (stopped at 0x" << std::hex << proc_->get_state()->pc
                    << ", target was 0x" << instruction_region_start_ << ")";
                last_error_ = oss.str();
                return false;
            }

            if (verbose_) {
                std::cout << "[SpikeEngine] Reached main after " << steps << " steps" << std::endl;
            }
        } else {
            if (verbose_) {
                std::cout << "[SpikeEngine] Entry point is already at main" << std::endl;
            }
        }

        initialized_ = true;
        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("Initialization exception: ") + e.what();
        std::cerr << "[SpikeEngine] Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        last_error_ = "Unknown exception during initialization";
        std::cerr << "[SpikeEngine] Unknown exception!" << std::endl;
        return false;
    }
}

uint64_t SpikeEngine::read_symbol_address(const std::string& symbol_name) {
    // Use objdump to read symbol table from ELF
    // Format: objdump -t <elf_file> | grep <symbol>
    // Output: "address  flags section size name"

    std::string cmd = "riscv64-unknown-elf-objdump -t " + elf_path_ + " 2>/dev/null | grep ' " + symbol_name + "$'";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return 0;
    }

    char buffer[256];
    uint64_t address = 0;

    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        // Parse first field (address in hex)
        char* endptr;
        address = strtoull(buffer, &endptr, 16);
    }

    pclose(pipe);
    return address;
}

bool SpikeEngine::find_nop_region() {
    // Find instruction region using ELF symbol table
    // Read the 'main' symbol address to locate the instruction region

    // Try to read 'main' symbol address from ELF
    uint64_t main_addr = read_symbol_address("main");

    if (main_addr == 0) {
        // Fallback: try common alternatives
        main_addr = read_symbol_address("_start");
        if (main_addr == 0) {
            last_error_ = "Failed to find 'main' or '_start' symbol in ELF";
            return false;
        }
    }

    // Set instruction region bounds
    instruction_region_start_ = main_addr;
    next_instruction_addr_ = main_addr;

    // Calculate available space (assuming worst case: all 4-byte instructions)
    // This provides enough space even if all instructions are standard 32-bit
    instruction_region_end_ = main_addr + num_instrs_ * 4;

    return true;
}

void SpikeEngine::set_checkpoint() {
    save_state(checkpoint_);
    checkpoint_valid_ = true;
}

void SpikeEngine::restore_checkpoint() {
    if (!checkpoint_valid_) {
        throw std::runtime_error("No valid checkpoint to restore");
    }
    restore_state(checkpoint_);
}

ExecutionResult SpikeEngine::execute_instruction(uint32_t machine_code,
                                                  const std::vector<int>& source_regs,
                                                  const std::vector<int>& dest_regs,
                                                  int64_t immediate) {
    if (!initialized_) {
        throw std::runtime_error("SpikeEngine not initialized");
    }

    // Detect instruction size (2 for compressed, 4 for standard)
    size_t instr_size = get_instruction_size(machine_code);

    // Check if we have enough space
    if (next_instruction_addr_ + instr_size > instruction_region_end_) {
        throw std::runtime_error("Out of instruction region space");
    }

    // Verify PC matches next instruction address
    uint64_t current_pc = get_pc();
    if (current_pc != next_instruction_addr_) {
        std::ostringstream oss;
        oss << "PC mismatch: expected 0x" << std::hex << next_instruction_addr_
            << " but got 0x" << current_pc << std::dec;
        throw std::runtime_error(oss.str());
    }

    // Get instruction address
    uint64_t instr_addr = next_instruction_addr_;

    // Write machine code with correct size
    if (!write_memory(instr_addr, machine_code, instr_size)) {
        throw std::runtime_error("Failed to write machine code to memory");
    }

    // STEP 1: Read source register values BEFORE execution
    // This ensures we capture the values that will be used by the instruction,
    // even if the destination register overlaps with source registers
    // (e.g., add x10, x10, x11 - we want the OLD value of x10, not the result)
    std::vector<uint64_t> source_values;
    for (int reg_idx : source_regs) {
        if (reg_idx >= 0 && reg_idx < 32) {
            source_values.push_back(get_xpr(reg_idx));
        }
    }

    // Add immediate if instruction has one (including immediate=0)
    if (immediate != IMMEDIATE_NOT_PRESENT) {
        source_values.push_back(static_cast<uint64_t>(immediate));
    }

    // STEP 2: Execute instruction
    if (!step_processor()) {
        std::ostringstream oss;
        oss << "Failed to execute instruction at 0x" << std::hex << instr_addr
            << std::dec << " - " << last_error_;
        throw std::runtime_error(oss.str());
    }

    // STEP 3: Read destination register values AFTER execution
    // These values are used for bug filtering in Python (e.g., checking if sc.w returned 1)
    std::vector<uint64_t> dest_values;
    for (int reg_idx : dest_regs) {
        if (reg_idx >= 0 && reg_idx < 32) {
            dest_values.push_back(get_xpr(reg_idx));
        }
    }

    // Update next instruction address to current PC
    // (Spike automatically updates PC based on instruction length)
    next_instruction_addr_ = get_pc();

    // Increment instruction index
    current_instr_index_++;

    // Return both source values (for XOR) and dest values (for bug filtering)
    return ExecutionResult(source_values, dest_values);
}

uint64_t SpikeEngine::get_xpr(int reg_index) const {
    if (!proc_ || reg_index < 0 || reg_index >= 32) {
        return 0;
    }
    return proc_->get_state()->XPR[reg_index];
}

uint64_t SpikeEngine::get_fpr(int reg_index) const {
    if (!proc_ || reg_index < 0 || reg_index >= 32) {
        return 0;
    }
    return proc_->get_state()->FPR[reg_index].v[0];
}

uint64_t SpikeEngine::get_pc() const {
    if (!proc_) {
        return 0;
    }
    return proc_->get_state()->pc;
}

//==============================================================================
// Private Helper Methods
//==============================================================================

bool SpikeEngine::write_memory(uint64_t addr, uint32_t code, size_t size) {
    try {
        mmu_t* mmu = proc_->get_mmu();

        if (size == 2) {
            // Compressed instruction: write only 16 bits
            uint16_t code_16 = static_cast<uint16_t>(code & 0xFFFF);
            mmu->store<uint16_t>(addr, code_16);
        } else if (size == 4) {
            // Standard instruction: write full 32 bits
            mmu->store<uint32_t>(addr, code);
        } else {
            last_error_ = "Invalid instruction size: " + std::to_string(size);
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        last_error_ = std::string("Memory write failed: ") + e.what();
        return false;
    }
}

uint32_t SpikeEngine::read_memory(uint64_t addr) {
    try {
        mmu_t* mmu = proc_->get_mmu();
        return mmu->load<uint32_t>(addr);
    } catch (const std::exception&) {
        return 0;
    }
}

bool SpikeEngine::step_processor() {
    try {
        // Execute one instruction
        proc_->step(1);
        return true;
    } catch (trap_t& t) {
        // Catch RISC-V traps (illegal instruction, access faults, etc.)
        std::ostringstream oss;
        oss << "Processor trap: " << t.name()
            << " (cause=" << t.cause() << ")";

        if (t.has_tval()) {
            oss << ", tval=0x" << std::hex << t.get_tval() << std::dec;
        }

        last_error_ = oss.str();
        return false;
    } catch (trap_debug_mode&) {
        // Debug mode entry (not an error in fuzzing context, just skip)
        last_error_ = "Processor entered debug mode";
        return false;
    } catch (const std::exception& e) {
        last_error_ = std::string("Processor step failed: ") + e.what();
        return false;
    } catch (...) {
        // Catch all other exceptions
        last_error_ = "Processor step failed: Caught an unknown exception!";
        return false;
    }
}

uint64_t SpikeEngine::compute_xor(const std::vector<uint64_t>& values) const {
    uint64_t result = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        result ^= (values[i] << i);
    }
    return result;
}

void SpikeEngine::save_state(Checkpoint& checkpoint) {
    if (!proc_) {
        throw std::runtime_error("No processor to save state from");
    }

    try {
        auto state = proc_->get_state();

        // Save general-purpose registers
        try {
            for (int i = 0; i < 32; ++i) {
                checkpoint.xpr[i] = state->XPR[i];
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to save XPR registers: ") + e.what());
        } catch (...) {
            throw std::runtime_error("Failed to save XPR registers: Unknown exception");
        }

        // Save floating-point registers
        try {
            for (int i = 0; i < 32; ++i) {
                checkpoint.fpr[i] = state->FPR[i].v[0];
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to save FPR registers: ") + e.what());
        } catch (...) {
            throw std::runtime_error("Failed to save FPR registers: Unknown exception");
        }

        // Save program counter
        checkpoint.pc = state->pc;

        // Save instruction index
        // This marks the execution position at checkpoint time
        checkpoint.instr_index = current_instr_index_;

        // Save next instruction address
        // This ensures PC and instruction placement remain synchronized after restore
        checkpoint.next_instruction_addr = next_instruction_addr_;

    } catch (const std::runtime_error&) {
        // Re-throw RuntimeError as-is (already has detailed message)
        throw;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to save checkpoint: ") + e.what());
    } catch (...) {
        throw std::runtime_error("Failed to save checkpoint: Unknown exception in outer block");
    }
}

void SpikeEngine::restore_state(const Checkpoint& checkpoint) {
    if (!proc_) {
        throw std::runtime_error("No processor to restore state to");
    }

    try {
        auto state = proc_->get_state();

        // Restore general-purpose registers
        for (int i = 0; i < 32; ++i) {
            state->XPR.write(i, checkpoint.xpr[i]);
        }

        // Restore floating-point registers
        for (int i = 0; i < 32; ++i) {
            freg_t freg_val;
            freg_val.v[0] = checkpoint.fpr[i];
            freg_val.v[1] = 0;
            state->FPR.write(i, freg_val);
        }

        // Restore program counter
        state->pc = checkpoint.pc;

        // Restore instruction index
        current_instr_index_ = checkpoint.instr_index;

        // Restore next instruction address
        next_instruction_addr_ = checkpoint.next_instruction_addr;

        // Memory Consistency Note:
        //   Instructions executed after the checkpoint may remain in memory,
        //   but this is safe because:
        //   1. PC and next_instruction_addr now point to the checkpoint position
        //   2. execute_instruction() will overwrite these stale instructions
        //   3. No execution will occur at addresses beyond checkpoint position

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to restore checkpoint: ") + e.what());
    } catch (...) {
        throw std::runtime_error("Failed to restore checkpoint: Unknown exception");
    }
}

} // namespace spike_engine

#include "spike_engine.h"
#include "state_query.h"
#include "checkpoint.h"
#include "../riscv/sim.h"
#include "../riscv/processor.h"
#include "../riscv/mmu.h"
#include "../riscv/cfg.h"
#include "../riscv/decode.h"
#include "../riscv/decode_macros.h"  // For wait_for_interrupt_t
#include "../riscv/trap.h"
#include "../riscv/encoding.h"
#include "../riscv/triggers.h"      // For triggers::matched_t
#include "../riscv/vector_unit.h"

#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cxxabi.h> // For abi::__cxa_current_exception_type

namespace spike_engine {

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
    , mem_region_start_(0)
    , mem_region_size_(0)
    , current_instr_index_(0)
    , initialized_(false)
    , last_execution_trapped_(false)
    , last_trap_handler_steps_(0)
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
        // create main memory for our ELF (0x80000000 - 0x90000000)
        std::vector<std::pair<reg_t, abstract_mem_t*>> mems;

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
            /*dtb_enabled=*/true,   // Enable boot ROM for consistent minstret with spike
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

        // Initialize state query interface
        state_query_ = std::make_unique<StateQuery>(proc_, sim_.get());

        // Initialize checkpoint manager
        checkpoint_manager_ = std::make_unique<CheckpointManager>(proc_, sim_.get());
        checkpoint_manager_->set_memory_region(mem_region_start_, mem_region_size_);

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

    // Find mem_region for checkpoint memory backup
    // This is the data region that can be modified by instructions
    uint64_t mem_region_addr = read_symbol_address("mem_region");
    uint64_t mem_region_end_addr = read_symbol_address("mem_region_end");

    if (mem_region_addr != 0 && mem_region_end_addr != 0) {
        mem_region_start_ = mem_region_addr;
        mem_region_size_ = mem_region_end_addr - mem_region_addr;
        if (verbose_) {
            std::cout << "[SpikeEngine] mem_region: 0x" << std::hex << mem_region_start_
                      << " - 0x" << (mem_region_start_ + mem_region_size_)
                      << " (size: " << std::dec << mem_region_size_ << " bytes)" << std::endl;
        }
    } else {
        // Fallback: use a default size based on template (8KB = 8192 bytes)
        // The mem_region typically starts right after .data section
        mem_region_addr = read_symbol_address("region_0");
        if (mem_region_addr != 0) {
            // region_0 is 32 bytes, mem_region follows
            mem_region_start_ = mem_region_addr + 32;
            mem_region_size_ = 8192;  // Default .mem_region size
        }
        if (verbose_) {
            std::cout << "[SpikeEngine] mem_region symbols not found, using fallback" << std::endl;
        }
    }

    return true;
}

void SpikeEngine::set_checkpoint() {
    if (!checkpoint_manager_) {
        throw std::runtime_error("CheckpointManager not initialized");
    }
    checkpoint_manager_->save(checkpoint_, current_instr_index_, next_instruction_addr_);
}

void SpikeEngine::restore_checkpoint() {
    if (!checkpoint_manager_) {
        throw std::runtime_error("CheckpointManager not initialized");
    }
    if (!checkpoint_.is_valid()) {
        throw std::runtime_error("No valid checkpoint to restore");
    }
    checkpoint_manager_->restore(checkpoint_, current_instr_index_, next_instruction_addr_);
}

size_t SpikeEngine::execute_sequence(
    const std::vector<uint32_t>& machine_codes,
    const std::vector<size_t>& sizes,
    size_t max_steps) {

    if (!initialized_) {
        throw std::runtime_error("SpikeEngine not initialized");
    }

    if (machine_codes.size() != sizes.size()) {
        throw std::runtime_error("machine_codes and sizes must have the same length");
    }

    if (machine_codes.empty()) {
        return 0;
    }

    // Calculate target PC (address after the last instruction)
    size_t total_size = std::accumulate(sizes.begin(), sizes.end(), size_t(0));
    uint64_t target_pc = next_instruction_addr_ + total_size;

    // Check if we have enough space
    if (target_pc > instruction_region_end_) {
        throw std::runtime_error("Out of instruction region space");
    }

    // Verify PC matches next instruction address
    uint64_t current_pc = proc_->get_state()->pc;
    if (current_pc != next_instruction_addr_) {
        std::ostringstream oss;
        oss << "PC mismatch: expected 0x" << std::hex << next_instruction_addr_
            << " but got 0x" << current_pc << std::dec;
        throw std::runtime_error(oss.str());
    }

    // Write all instructions to memory
    uint64_t write_addr = next_instruction_addr_;
    for (size_t i = 0; i < machine_codes.size(); ++i) {
        if (!write_memory(write_addr, machine_codes[i], sizes[i])) {
            throw std::runtime_error("Failed to write instruction to memory: " + last_error_);
        }
        write_addr += sizes[i];
    }

    // Execute until PC reaches target address
    // This handles all cases uniformly:
    // - Sequential execution: PC advances to target_pc after each step
    // - Trap handling: Spike handles traps internally, mret returns to next instruction
    // - Forward jumps: PC jumps directly to target
    // - Backward loops: PC loops until branch falls through to target_pc
    size_t steps = 0;
    while (proc_->get_state()->pc != target_pc && steps < max_steps) {
        try {
            proc_->step(1);
        } catch (wait_for_interrupt_t&) {
            // WFI instruction - PC already updated, continue execution
        }
        steps++;
    }

    if (steps >= max_steps) {
        std::ostringstream oss;
        oss << "Execution exceeded max_steps (" << max_steps << "), PC=0x"
            << std::hex << proc_->get_state()->pc << ", target_pc=0x" << target_pc;
        throw std::runtime_error(oss.str());
    }

    // Update state
    next_instruction_addr_ = target_pc;
    current_instr_index_ += machine_codes.size();

    // Record trap information (inferred from extra steps)
    size_t expected_steps = machine_codes.size();
    if (steps > expected_steps) {
        last_execution_trapped_ = true;
        last_trap_handler_steps_ = steps - expected_steps;
    } else {
        last_execution_trapped_ = false;
        last_trap_handler_steps_ = 0;
    }

    return machine_codes.size();
}

//==============================================================================
// Private Helper Methods
//==============================================================================

bool SpikeEngine::write_memory(uint64_t addr, uint32_t code, size_t size) {
    try {
        // Use debug_mmu to bypass PMP/PMA checks so we can always inject instructions
        // regardless of the current processor state (e.g. if previous instructions
        // messed up PMP settings).
        // mmu_t* mmu = proc_->get_mmu();
        mmu_t* mmu = sim_->debug_mmu;

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

        // CRITICAL: Flush TLB and instruction cache after writing to instruction memory.
        // Even though we wrote via debug_mmu, the processor's MMU/ICache might still
        // hold stale entries.
        proc_->get_mmu()->flush_tlb();

        return true;
    } catch (const std::exception& e) {
        last_error_ = std::string("Memory write failed: ") + e.what();
        return false;
    } catch (trap_t& t) {
        std::ostringstream oss;
        oss << "Memory write failed: Trap exception: " << t.name() << " (cause=" << t.cause() << ")";
        last_error_ = oss.str();
        return false;
    } catch (triggers::matched_t& t) {
        std::ostringstream oss;
        oss << "Memory write failed: Trigger matched: address=0x" << std::hex << t.address << std::dec;
        last_error_ = oss.str();
        return false;
    } catch (...) {
        std::ostringstream oss;
        oss << "Memory write failed: Unknown exception";
        // Advanced: try to get the real type name
        std::type_info* t = abi::__cxa_current_exception_type();
        if (t) {
            int status;
            char* demangled = abi::__cxa_demangle(t->name(), 0, 0, &status);
            oss << " (Type: " << (demangled ? demangled : t->name()) << ")";
            if (demangled) free(demangled);
        } else {
            oss << " (truly unknown type)";
        }
        last_error_ = oss.str();
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
    } catch (wait_for_interrupt_t&) {
        // WFI instruction
        last_error_ = "Processor hit WFI (wait_for_interrupt_t)";
        return false;
    } catch (triggers::matched_t& t) {
        // Hardware trigger matched
        std::ostringstream oss;
        oss << "Processor hit Trigger (triggers::matched_t): address=0x" << std::hex << t.address << std::dec;
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
        // Catch all other exceptions - try to get more info
        std::exception_ptr eptr = std::current_exception();
        std::ostringstream oss;
        oss << "Processor step failed: Caught an unknown exception!";

        // Try to rethrow and catch as different types for more info
        if (eptr) {
            try {
                std::rethrow_exception(eptr);
            } catch (const std::exception& e) {
                oss << " (std::exception: " << e.what() << ")";
            } catch (const char* msg) {
                oss << " (C-string: " << msg << ")";
            } catch (int e) {
                oss << " (int: " << e << ")";
            } catch (...) {
                oss << " (truly unknown type)";
            }
        }

        last_error_ = oss.str();
        return false;
    }
}

} // namespace spike_engine

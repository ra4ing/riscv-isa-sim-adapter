// Copyright (c) 2024-2025 DiveFuzz Project
// SPDX-License-Identifier: Mulan PSL v2

#include "checkpoint.h"
#include "../riscv/processor.h"
#include "../riscv/mmu.h"
#include "../riscv/sim.h"
#include "../riscv/encoding.h"
#include "../riscv/csrs.h"
#include "../riscv/vector_unit.h"

#include <stdexcept>
#include <chrono>
#include <cstring>

namespace spike_engine {

//==============================================================================
// Checkpoint Implementation
//==============================================================================

Checkpoint::Checkpoint()
    : xpr(32, 0)
    , fpr(32, 0)
    , fpr_v1(32, 0)
    , pc(0)
    , instr_index(0)
    , next_instruction_addr(0)
    , prv(3)  // Default to M-mode
    , v(false)
    , debug_mode(false)
    , prev_prv(3)
    , prev_v(false)
    , serialized(true)
    , elp(0)
    , single_step(0)
    , critical_error(false)
    , valid(false)
    , timestamp(0)
    , id(0)
{
}

void Checkpoint::clear() {
    std::fill(xpr.begin(), xpr.end(), 0);
    std::fill(fpr.begin(), fpr.end(), 0);
    std::fill(fpr_v1.begin(), fpr_v1.end(), 0);
    pc = 0;
    instr_index = 0;
    next_instruction_addr = 0;
    mem_region_backup.clear();
    prv = 3;
    v = false;
    debug_mode = false;
    csr_values.clear();
    reservation.clear();
    prev_prv = 3;
    prev_v = false;
    vector_regfile.clear();
    serialized = true;
    elp = 0;
    single_step = 0;
    critical_error = false;
    valid = false;
    timestamp = 0;
}

size_t Checkpoint::memory_usage() const {
    size_t usage = sizeof(Checkpoint);
    usage += xpr.capacity() * sizeof(uint64_t);
    usage += fpr.capacity() * sizeof(uint64_t);
    usage += fpr_v1.capacity() * sizeof(uint64_t);
    usage += mem_region_backup.capacity();
    usage += csr_values.size() * (sizeof(uint64_t) * 2 + 32);  // Approximate map overhead
    usage += vector_regfile.capacity();
    return usage;
}

//==============================================================================
// CheckpointManager Implementation
//==============================================================================

CheckpointManager::CheckpointManager(processor_t* proc, sim_t* sim)
    : proc_(proc)
    , sim_(sim)
    , mem_region_start_(0)
    , mem_region_size_(0)
    , next_checkpoint_id_(1)
{
}

void CheckpointManager::set_memory_region(uint64_t start, size_t size) {
    mem_region_start_ = start;
    mem_region_size_ = size;
}

//==============================================================================
// Save/Restore Operations
//==============================================================================

void CheckpointManager::save(Checkpoint& checkpoint, size_t instr_index, uint64_t next_instr_addr) {
    if (!proc_) {
        throw std::runtime_error("No processor to save state from");
    }

    // Set metadata
    checkpoint.instr_index = instr_index;
    checkpoint.next_instruction_addr = next_instr_addr;
    checkpoint.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Save all state
    save_registers(checkpoint);
    save_privilege_state(checkpoint);
    save_csrs(checkpoint);
    save_memory(checkpoint);
    save_extended_state(checkpoint);

    checkpoint.valid = true;
}

void CheckpointManager::restore(const Checkpoint& checkpoint, size_t& instr_index, uint64_t& next_instr_addr) {
    if (!proc_) {
        throw std::runtime_error("No processor to restore state to");
    }

    if (!checkpoint.valid) {
        throw std::runtime_error("Cannot restore from invalid checkpoint");
    }

    // Restore privilege state FIRST (CSR access depends on privilege)
    restore_privilege_state(checkpoint);

    // Restore CSRs
    restore_csrs(checkpoint);

    // Restore registers
    restore_registers(checkpoint);

    // Restore memory
    restore_memory(checkpoint);

    // Restore extended state
    restore_extended_state(checkpoint);

    // Flush TLB and instruction cache
    proc_->get_mmu()->flush_tlb();

    // Return execution position
    instr_index = checkpoint.instr_index;
    next_instr_addr = checkpoint.next_instruction_addr;
}

//==============================================================================
// Multi-Checkpoint Operations
//==============================================================================

uint32_t CheckpointManager::create_checkpoint(size_t instr_index, uint64_t next_instr_addr) {
    uint32_t id = next_checkpoint_id_++;
    Checkpoint& cp = checkpoints_[id];
    cp.id = id;
    save(cp, instr_index, next_instr_addr);
    return id;
}

Checkpoint* CheckpointManager::get_checkpoint(uint32_t id) {
    auto it = checkpoints_.find(id);
    return (it != checkpoints_.end()) ? &it->second : nullptr;
}

const Checkpoint* CheckpointManager::get_checkpoint(uint32_t id) const {
    auto it = checkpoints_.find(id);
    return (it != checkpoints_.end()) ? &it->second : nullptr;
}

bool CheckpointManager::restore_checkpoint(uint32_t id, size_t& instr_index, uint64_t& next_instr_addr) {
    const Checkpoint* cp = get_checkpoint(id);
    if (!cp) {
        return false;
    }
    restore(*cp, instr_index, next_instr_addr);
    return true;
}

bool CheckpointManager::delete_checkpoint(uint32_t id) {
    return checkpoints_.erase(id) > 0;
}

void CheckpointManager::clear_all_checkpoints() {
    checkpoints_.clear();
}

std::vector<uint32_t> CheckpointManager::get_checkpoint_ids() const {
    std::vector<uint32_t> ids;
    ids.reserve(checkpoints_.size());
    for (const auto& [id, _] : checkpoints_) {
        ids.push_back(id);
    }
    return ids;
}

//==============================================================================
// Internal Helper Methods
//==============================================================================

void CheckpointManager::save_registers(Checkpoint& checkpoint) {
    auto state = proc_->get_state();

    // Save general-purpose registers
    for (int i = 0; i < 32; ++i) {
        checkpoint.xpr[i] = state->XPR[i];
    }

    // Save floating-point registers (both v[0] and v[1])
    for (int i = 0; i < 32; ++i) {
        checkpoint.fpr[i] = state->FPR[i].v[0];
        checkpoint.fpr_v1[i] = state->FPR[i].v[1];
    }

    // Save program counter
    checkpoint.pc = state->pc;
}

void CheckpointManager::restore_registers(const Checkpoint& checkpoint) {
    auto state = proc_->get_state();

    // Restore general-purpose registers
    for (int i = 0; i < 32; ++i) {
        state->XPR.write(i, checkpoint.xpr[i]);
    }

    // Restore floating-point registers (both v[0] and v[1])
    for (int i = 0; i < 32; ++i) {
        freg_t freg_val;
        freg_val.v[0] = checkpoint.fpr[i];
        freg_val.v[1] = checkpoint.fpr_v1[i];
        state->FPR.write(i, freg_val);
    }

    // Restore program counter
    state->pc = checkpoint.pc;
}

void CheckpointManager::save_privilege_state(Checkpoint& checkpoint) {
    auto state = proc_->get_state();

    checkpoint.prv = state->prv;
    checkpoint.v = state->v;
    checkpoint.debug_mode = state->debug_mode;
}

void CheckpointManager::restore_privilege_state(const Checkpoint& checkpoint) {
    auto state = proc_->get_state();

    state->prv = checkpoint.prv;
    state->v = checkpoint.v;
    state->debug_mode = checkpoint.debug_mode;
}

void CheckpointManager::save_csrs(Checkpoint& checkpoint) {
    auto state = proc_->get_state();

    checkpoint.csr_values.clear();

    // Iterate through all CSRs and save their values
    for (const auto& [addr, csr] : state->csrmap) {
        if (csr) {
            try {
                checkpoint.csr_values[addr] = csr->read();
            } catch (...) {
                // Skip CSRs that throw exceptions on read
            }
        }
    }
}

void CheckpointManager::restore_csrs(const Checkpoint& checkpoint) {
    auto state = proc_->get_state();

    // Helper lambda for restoring a CSR
    auto restore_csr = [&checkpoint](reg_t addr, const csr_t_p& csr) {
        if (csr && checkpoint.csr_values.count(addr)) {
            try {
                csr->write(checkpoint.csr_values.at(addr));
            } catch (...) {
                // Silently skip CSRs that fail to write
            }
        }
    };

    // Machine mode trap handling CSRs
    restore_csr(0x300, state->mstatus);
    restore_csr(0x301, state->misa);
    restore_csr(0x302, state->medeleg);
    restore_csr(0x303, state->mideleg);
    restore_csr(0x304, state->mie);
    restore_csr(0x305, state->mtvec);
    restore_csr(0x306, state->mcounteren);
    restore_csr(0x310, state->mstatush);
    restore_csr(0x320, state->mcountinhibit);
    restore_csr(0x340, state->csrmap.count(0x340) ? state->csrmap.at(0x340) : nullptr);
    restore_csr(0x341, state->mepc);
    restore_csr(0x342, state->mcause);
    restore_csr(0x343, state->mtval);
    restore_csr(0x344, state->mip);

    // Supervisor mode CSRs
    restore_csr(0x100, state->sstatus);
    restore_csr(0x104, state->nonvirtual_sie);
    restore_csr(0x105, state->stvec);
    restore_csr(0x106, state->scounteren);
    restore_csr(0x140, state->csrmap.count(0x140) ? state->csrmap.at(0x140) : nullptr);
    restore_csr(0x141, state->sepc);
    restore_csr(0x142, state->scause);
    restore_csr(0x143, state->stval);
    restore_csr(0x144, state->nonvirtual_sip);
    restore_csr(0x180, state->satp);

    // Floating-point CSRs (need special handling for dirty state)
    {
        reg_t current_mstatus = state->mstatus->read();
        reg_t temp_mstatus = (current_mstatus & ~MSTATUS_FS) | MSTATUS_FS;
        state->mstatus->write(temp_mstatus);

        restore_csr(0x001, state->fflags);
        restore_csr(0x002, state->frm);
    }

    // Vector extension CSRs
    if (proc_->VU.VLEN > 0) {
        reg_t current_mstatus = state->mstatus->read();
        reg_t temp_mstatus = (current_mstatus & ~MSTATUS_VS) | MSTATUS_VS;
        state->mstatus->write(temp_mstatus);

        if (proc_->VU.vstart && checkpoint.csr_values.count(CSR_VSTART)) {
            proc_->VU.vstart->write_raw(checkpoint.csr_values.at(CSR_VSTART));
        }
        if (proc_->VU.vxrm && checkpoint.csr_values.count(CSR_VXRM)) {
            proc_->VU.vxrm->write_raw(checkpoint.csr_values.at(CSR_VXRM));
        }
        if (proc_->VU.vl && checkpoint.csr_values.count(CSR_VL)) {
            proc_->VU.vl->write_raw(checkpoint.csr_values.at(CSR_VL));
        }
        if (proc_->VU.vtype && checkpoint.csr_values.count(CSR_VTYPE)) {
            proc_->VU.vtype->write_raw(checkpoint.csr_values.at(CSR_VTYPE));
        }
        if (proc_->VU.vxsat && checkpoint.csr_values.count(CSR_VXSAT)) {
            try {
                proc_->VU.vxsat->write(checkpoint.csr_values.at(CSR_VXSAT));
            } catch (...) {}
        }
    }

    // Environment configuration CSRs
    restore_csr(0x30A, state->menvcfg);
    restore_csr(0x10A, state->senvcfg);

    // Hypervisor CSRs
    restore_csr(0x600, state->hstatus);
    restore_csr(0x602, state->hedeleg);
    restore_csr(0x603, state->hideleg);
    restore_csr(0x604, state->hvip);
    restore_csr(0x605, state->htimedelta);
    restore_csr(0x606, state->hcounteren);
    restore_csr(0x60A, state->henvcfg);
    restore_csr(0x643, state->htval);
    restore_csr(0x64A, state->htinst);
    restore_csr(0x680, state->hgatp);

    // VS mode CSRs
    restore_csr(0x200, state->vsstatus);
    restore_csr(0x205, state->vstvec);
    restore_csr(0x241, state->vsepc);
    restore_csr(0x242, state->vscause);
    restore_csr(0x243, state->vstval);
    restore_csr(0x280, state->vsatp);

    // Debug CSRs
    restore_csr(0x7B0, state->dcsr);
    restore_csr(0x7B1, state->dpc);

    // Trigger CSRs
    restore_csr(0x7A0, state->tselect);
    restore_csr(0x7A2, state->tdata2);
    restore_csr(0x7A5, state->tcontrol);
    restore_csr(0x5A8, state->scontext);
    restore_csr(0x7A8, state->mcontext);

    // Counter CSRs (need special handling)
    if (state->mcycle) {
        state->mcycle->bump(0);
        restore_csr(0xB00, state->mcycle);
    }
    if (state->minstret) {
        state->minstret->bump(0);
        restore_csr(0xB02, state->minstret);
    }

    // PMP CSRs
    for (int i = 0; i < 64 && state->pmpaddr[i]; ++i) {
        restore_csr(0x3B0 + i, state->pmpaddr[i]);
    }
    for (reg_t addr = 0x3A0; addr <= 0x3AF; ++addr) {
        if (state->csrmap.count(addr)) {
            restore_csr(addr, state->csrmap.at(addr));
        }
    }

    // Re-restore mstatus/sstatus to override dirty bits
    restore_csr(0x300, state->mstatus);
    restore_csr(0x100, state->sstatus);
}

void CheckpointManager::save_memory(Checkpoint& checkpoint) {
    if (mem_region_size_ == 0 || mem_region_start_ == 0 || !sim_) {
        return;
    }

    checkpoint.mem_region_backup.resize(mem_region_size_);

    for (size_t i = 0; i < mem_region_size_; i += 8) {
        uint64_t addr = mem_region_start_ + i;
        for (size_t j = 0; j < 8 && (i + j) < mem_region_size_; ++j) {
            checkpoint.mem_region_backup[i + j] = sim_->debug_mmu->load<uint8_t>(addr + j);
        }
    }
}

void CheckpointManager::restore_memory(const Checkpoint& checkpoint) {
    if (checkpoint.mem_region_backup.empty() || mem_region_size_ == 0 || !sim_) {
        return;
    }

    for (size_t i = 0; i < checkpoint.mem_region_backup.size(); ++i) {
        sim_->debug_mmu->store<uint8_t>(mem_region_start_ + i, checkpoint.mem_region_backup[i]);
    }
}

void CheckpointManager::save_extended_state(Checkpoint& checkpoint) {
    if (!proc_) {
        return;
    }

    auto state = proc_->get_state();

    // Save reservation state
    mmu_t* mmu = proc_->get_mmu();
    if (mmu) {
        checkpoint.reservation.address = mmu->get_load_reservation_address();
        checkpoint.reservation.valid = (checkpoint.reservation.address != ReservationState::INVALID_ADDR);
    }

    // Save privilege transition state
    checkpoint.prev_prv = state->prev_prv;
    checkpoint.prev_v = state->prev_v;

    // Save serialized flag
    checkpoint.serialized = state->serialized;

    // Save ELP (Expected Landing Pad) state for Zicfilp extension
    checkpoint.elp = static_cast<uint8_t>(state->elp);

    // Save single step state (debug mode)
    checkpoint.single_step = static_cast<uint8_t>(state->single_step);

    // Save critical error flag
    checkpoint.critical_error = state->critical_error;

    // Save vector register file if V extension is enabled
    if (proc_->VU.VLEN > 0 && proc_->VU.reg_file) {
        size_t vreg_size = 32 * (proc_->VU.VLEN / 8);
        checkpoint.vector_regfile.resize(vreg_size);
        std::memcpy(checkpoint.vector_regfile.data(), proc_->VU.reg_file, vreg_size);
    }
}

void CheckpointManager::restore_extended_state(const Checkpoint& checkpoint) {
    if (!proc_) {
        return;
    }

    auto state = proc_->get_state();

    // Restore privilege transition state
    state->prev_prv = checkpoint.prev_prv;
    state->prev_v = checkpoint.prev_v;

    // Restore serialized flag
    state->serialized = checkpoint.serialized;

    // Restore ELP (Expected Landing Pad) state for Zicfilp extension
    state->elp = static_cast<elp_t>(checkpoint.elp);

    // Restore single step state (debug mode)
    state->single_step = static_cast<decltype(state->single_step)>(checkpoint.single_step);

    // Restore critical error flag
    state->critical_error = checkpoint.critical_error;

    // Restore reservation state
    mmu_t* mmu = proc_->get_mmu();
    if (mmu) {
        if (checkpoint.reservation.valid) {
            mmu->set_load_reservation_address(checkpoint.reservation.address);
        } else {
            mmu->yield_load_reservation();
        }
    }

    // Restore vector register file if present
    if (!checkpoint.vector_regfile.empty() && proc_->VU.VLEN > 0 && proc_->VU.reg_file) {
        size_t vreg_size = 32 * (proc_->VU.VLEN / 8);
        if (checkpoint.vector_regfile.size() == vreg_size) {
            std::memcpy(proc_->VU.reg_file, checkpoint.vector_regfile.data(), vreg_size);
        }
    }
}

} // namespace spike_engine

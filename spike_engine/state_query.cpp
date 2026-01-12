// Copyright (c) 2024-2025 DiveFuzz Project
// SPDX-License-Identifier: Mulan PSL v2

#include "state_query.h"
#include "../riscv/processor.h"
#include "../riscv/mmu.h"
#include "../riscv/sim.h"
#include "../riscv/vector_unit.h"

#include <stdexcept>
#include <cstring>

namespace spike_engine {

//==============================================================================
// StateQuery Implementation
//==============================================================================

StateQuery::StateQuery(processor_t* proc, sim_t* sim)
    : proc_(proc)
    , sim_(sim)
{
}

//==============================================================================
// Basic Register Queries
//==============================================================================

uint64_t StateQuery::get_xpr(int reg_index) const {
    if (!proc_ || reg_index < 0 || reg_index >= 32) {
        return 0;
    }
    return proc_->get_state()->XPR[reg_index];
}

std::vector<uint64_t> StateQuery::get_all_xpr() const {
    std::vector<uint64_t> result(32, 0);
    if (!proc_) {
        return result;
    }
    for (int i = 0; i < 32; ++i) {
        result[i] = proc_->get_state()->XPR[i];
    }
    return result;
}

uint64_t StateQuery::get_fpr(int reg_index) const {
    if (!proc_ || reg_index < 0 || reg_index >= 32) {
        return 0;
    }
    return proc_->get_state()->FPR[reg_index].v[0];
}

std::vector<uint64_t> StateQuery::get_all_fpr() const {
    std::vector<uint64_t> result(32, 0);
    if (!proc_) {
        return result;
    }
    for (int i = 0; i < 32; ++i) {
        result[i] = proc_->get_state()->FPR[i].v[0];
    }
    return result;
}

uint64_t StateQuery::get_pc() const {
    if (!proc_) {
        return 0;
    }
    return proc_->get_state()->pc;
}

//==============================================================================
// CSR Queries
//==============================================================================

uint64_t StateQuery::get_csr(uint64_t csr_addr) const {
    if (!proc_) {
        return 0;
    }
    try {
        auto state = proc_->get_state();
        auto it = state->csrmap.find(csr_addr);
        if (it != state->csrmap.end() && it->second) {
            return it->second->read();
        }
    } catch (...) {
        // CSR read may throw exception, return 0
    }
    return 0;
}

std::map<uint64_t, uint64_t> StateQuery::get_all_csrs() const {
    std::map<uint64_t, uint64_t> result;
    if (!proc_) {
        return result;
    }
    try {
        auto state = proc_->get_state();
        for (const auto& [addr, csr] : state->csrmap) {
            if (csr) {
                try {
                    result[addr] = csr->read();
                } catch (...) {
                    // Skip CSRs that throw on read
                }
            }
        }
    } catch (...) {
        // Return partial result on error
    }
    return result;
}

//==============================================================================
// Memory Queries
//==============================================================================

std::vector<uint8_t> StateQuery::read_mem(uint64_t addr, size_t size) const {
    std::vector<uint8_t> result(size, 0);
    if (!sim_ || !sim_->debug_mmu) {
        return result;
    }
    try {
        for (size_t i = 0; i < size; ++i) {
            result[i] = sim_->debug_mmu->load<uint8_t>(addr + i);
        }
    } catch (...) {
        // Return partial result on error
    }
    return result;
}

//==============================================================================
// Commit Log Queries
//==============================================================================

CommitLog StateQuery::get_commit_log() const {
    CommitLog log;

    if (!proc_) {
        return log;
    }

    auto state = proc_->get_state();

    // Copy register writes
    // state->log_reg_write is typically a map<int, commit_log_reg_t::value_type>
    // Format: { reg_num -> { value, is_fp } }
    for (const auto& [reg_num, entry] : state->log_reg_write) {
        RegisterWrite rw;
        rw.reg_num = static_cast<uint16_t>(reg_num);
        // The value is stored in the entry (freg_t for FP, reg_t for integer)
        // We extract the 64-bit value
        rw.value = entry.v[0];  // Main 64-bit value
        log.reg_writes.push_back(rw);
    }

    // Copy memory reads
    // state->log_mem_read is vector of { addr, value, size }
    for (const auto& entry : state->log_mem_read) {
        MemoryAccess ma;
        ma.addr = std::get<0>(entry);
        ma.value = std::get<1>(entry);
        ma.size = static_cast<uint8_t>(std::get<2>(entry));
        log.mem_reads.push_back(ma);
    }

    // Copy memory writes
    for (const auto& entry : state->log_mem_write) {
        MemoryAccess ma;
        ma.addr = std::get<0>(entry);
        ma.value = std::get<1>(entry);
        ma.size = static_cast<uint8_t>(std::get<2>(entry));
        log.mem_writes.push_back(ma);
    }

    // Copy execution context
    log.inst_priv = state->last_inst_priv;
    log.inst_xlen = state->last_inst_xlen;
    log.inst_flen = state->last_inst_flen;

    return log;
}

void StateQuery::clear_commit_log() {
    if (!proc_) {
        return;
    }

    auto state = proc_->get_state();
    state->log_reg_write.clear();
    state->log_mem_read.clear();
    state->log_mem_write.clear();
}

//==============================================================================
// Privilege State Queries
//==============================================================================

PrivilegeState StateQuery::get_privilege_state() const {
    PrivilegeState ps;

    if (!proc_) {
        return ps;
    }

    auto state = proc_->get_state();

    ps.prv = state->prv;
    ps.prev_prv = state->prev_prv;
    ps.prv_changed = state->prv_changed;
    ps.v = state->v;
    ps.prev_v = state->prev_v;
    ps.v_changed = state->v_changed;
    ps.debug_mode = state->debug_mode;

    return ps;
}

bool StateQuery::did_privilege_change() const {
    if (!proc_) {
        return false;
    }

    auto state = proc_->get_state();
    return state->prv_changed || state->v_changed;
}

//==============================================================================
// Debug State Queries
//==============================================================================

DebugState StateQuery::get_debug_state() const {
    DebugState ds;

    if (!proc_) {
        return ds;
    }

    auto state = proc_->get_state();

    ds.debug_mode = state->debug_mode;
    ds.single_step = static_cast<uint8_t>(state->single_step);
    ds.critical_error = state->critical_error;
    ds.elp = static_cast<uint8_t>(state->elp);
    ds.serialized = state->serialized;

    return ds;
}

bool StateQuery::is_debug_mode() const {
    if (!proc_) {
        return false;
    }
    return proc_->get_state()->debug_mode;
}

bool StateQuery::is_single_stepping() const {
    if (!proc_) {
        return false;
    }
    auto state = proc_->get_state();
    return state->single_step != state->STEP_NONE;
}

bool StateQuery::has_critical_error() const {
    if (!proc_) {
        return false;
    }
    return proc_->get_state()->critical_error;
}

uint8_t StateQuery::get_elp() const {
    if (!proc_) {
        return 0;
    }
    return static_cast<uint8_t>(proc_->get_state()->elp);
}

//==============================================================================
// Trap/Exception Queries
//==============================================================================

TrapInfo StateQuery::get_last_trap_info() const {
    return last_trap_;
}

void StateQuery::set_trap_info(const TrapInfo& info) {
    last_trap_ = info;
}

void StateQuery::clear_trap_info() {
    last_trap_.clear();
}

//==============================================================================
// Vector Unit Queries
//==============================================================================

VectorState StateQuery::get_vector_state(bool include_regfile) const {
    VectorState vs;

    if (!proc_) {
        return vs;
    }

    // Check if V extension is enabled
    if (proc_->VU.VLEN == 0) {
        return vs;  // V extension not enabled
    }

    const auto& vu = proc_->VU;

    // Read CSR values
    vs.vl = vu.vl ? vu.vl->read() : 0;
    vs.vtype = vu.vtype ? vu.vtype->read() : 0;
    vs.vstart = vu.vstart ? vu.vstart->read() : 0;
    vs.vxsat = vu.vxsat ? vu.vxsat->read() : 0;
    vs.vxrm = vu.vxrm ? vu.vxrm->read() : 0;
    vs.vlenb = vu.vlenb;

    // Read internal state
    vs.vlmax = vu.vlmax;
    vs.vsew = vu.vsew;
    vs.vflmul = vu.vflmul;
    vs.vma = vu.vma;
    vs.vta = vu.vta;
    vs.vill = vu.vill;
    vs.VLEN = vu.VLEN;
    vs.ELEN = vu.ELEN;

    // Optionally copy the entire vector register file
    if (include_regfile && vu.reg_file) {
        // Vector register file size: 32 registers * VLEN/8 bytes each
        size_t vreg_size = 32 * (vu.VLEN / 8);
        vs.vreg_file.resize(vreg_size);
        std::memcpy(vs.vreg_file.data(), vu.reg_file, vreg_size);
    }

    return vs;
}

bool StateQuery::is_vector_enabled() const {
    if (!proc_) {
        return false;
    }
    return proc_->VU.VLEN > 0;
}

//==============================================================================
// Load Reservation Queries
//==============================================================================

ReservationState StateQuery::get_reservation_state() const {
    ReservationState rs;

    if (!proc_) {
        return rs;
    }

    mmu_t* mmu = proc_->get_mmu();
    if (!mmu) {
        return rs;
    }

    // Get the load reservation address from MMU
    rs.address = mmu->get_load_reservation_address();
    rs.valid = (rs.address != ReservationState::INVALID_ADDR);

    return rs;
}

bool StateQuery::has_reservation() const {
    ReservationState rs = get_reservation_state();
    return rs.valid;
}

void StateQuery::clear_reservation() {
    if (!proc_) {
        return;
    }

    mmu_t* mmu = proc_->get_mmu();
    if (mmu) {
        mmu->yield_load_reservation();
    }
}

} // namespace spike_engine

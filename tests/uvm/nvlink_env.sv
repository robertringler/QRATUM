// SPDX-License-Identifier: Apache-2.0
`include "uvm_macros.svh"
import uvm_pkg::*;

class nvlink_env extends uvm_env;
    `uvm_component_utils(nvlink_env)

    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
        phase.raise_objection(this);
        `uvm_info("NVLINK", "Running synthetic coherency sequence", UVM_MEDIUM)
        #10ns;
        phase.drop_objection(this);
    endtask
endclass

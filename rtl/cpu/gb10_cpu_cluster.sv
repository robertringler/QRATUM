// SPDX-License-Identifier: Apache-2.0
// Synthetic 72-core CPU cluster model with simplified execution pipeline.

module gb10_cpu_cluster #(
    parameter int CORE_COUNT = 72,
    parameter int XLEN = 64
) (
    input  logic         clk_i,
    input  logic         rst_ni,
    output logic [63:0]  perf_counter_o,
    input  logic [63:0]  workload_hint_i
);

    typedef struct packed {
        logic active;
        logic [XLEN-1:0] pc;
        logic [XLEN-1:0] regs;
    } core_state_t;

    core_state_t core_state [CORE_COUNT];
    logic [63:0] perf_counter;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            perf_counter <= '0;
            for (int i = 0; i < CORE_COUNT; i++) begin
                core_state[i].active <= 1'b0;
                core_state[i].pc <= '0;
                core_state[i].regs <= '0;
            end
        end else begin
            perf_counter <= perf_counter + CORE_COUNT;
            for (int i = 0; i < CORE_COUNT; i++) begin
                core_state[i].active <= 1'b1;
                core_state[i].pc <= core_state[i].pc + workload_hint_i[5:0];
                core_state[i].regs <= core_state[i].regs + workload_hint_i;
            end
        end
    end

    assign perf_counter_o = perf_counter;

endmodule

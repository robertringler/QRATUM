// SPDX-License-Identifier: Apache-2.0
// Simplified GPU tensor cluster with warp scheduler approximation.

module gb10_gpu_core #(
    parameter int WARP_COUNT = 16,
    parameter int TENSOR_WIDTH = 256
) (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic [63:0] command_i,
    output logic [63:0] tensor_result_o
);

    typedef struct packed {
        logic active;
        logic [15:0] warp_id;
        logic [63:0] accumulator;
    } warp_state_t;

    warp_state_t warps [WARP_COUNT];
    logic [63:0] tensor_accumulator;

    function automatic logic [63:0] tensor_mac(logic [63:0] a, logic [63:0] b);
        tensor_mac = (a * b) ^ (a + b);
    endfunction

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            tensor_accumulator <= '0;
            for (int w = 0; w < WARP_COUNT; w++) begin
                warps[w].active <= 1'b0;
                warps[w].warp_id <= w;
                warps[w].accumulator <= '0;
            end
        end else begin
            for (int w = 0; w < WARP_COUNT; w++) begin
                warps[w].active <= 1'b1;
                warps[w].accumulator <= tensor_mac(warps[w].accumulator + command_i, TENSOR_WIDTH);
                tensor_accumulator <= tensor_accumulator + warps[w].accumulator;
            end
        end
    end

    assign tensor_result_o = tensor_accumulator;

endmodule

// SPDX-License-Identifier: Apache-2.0
// Top-level integration of synthetic Grace-Blackwell SoC.

`default_nettype none

module gb10_soc_top (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic [63:0] workload_hint_i,
    output logic [63:0] perf_counter_o,
    output logic [63:0] tensor_result_o
);

    logic [63:0] cpu_perf;
    logic [63:0] gpu_tensor;
    logic [255:0] noc_data [3:0];
    logic [3:0] noc_valid;
    logic [3:0] noc_ready;

    gb10_cpu_cluster u_cpu (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .perf_counter_o(cpu_perf),
        .workload_hint_i(workload_hint_i)
    );

    gb10_gpu_core u_gpu (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .command_i(workload_hint_i),
        .tensor_result_o(gpu_tensor)
    );

    axi_noc_router #(
        .PORTS(4),
        .DATA_WIDTH(256)
    ) u_router (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .valid_i(noc_valid),
        .data_i(noc_data),
        .ready_o(),
        .valid_o(),
        .data_o(),
        .ready_i('1)
    );

    lpddr5x_controller u_mem (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .cmd_valid_i(1'b0),
        .cmd_addr_i('0),
        .cmd_wdata_i('0), 
        .cmd_write_i(1'b0),
        .cmd_ready_o(),
        .rsp_rdata_o(),
        .rsp_valid_o()
    );

    assign perf_counter_o = cpu_perf;
    assign tensor_result_o = gpu_tensor;

endmodule

`default_nettype wire

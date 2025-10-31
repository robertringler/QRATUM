// SPDX-License-Identifier: Apache-2.0
// Minimalistic LPDDR5x memory controller behavioral model.

module lpddr5x_controller #(
    parameter int ADDR_WIDTH = 32,
    parameter int DATA_WIDTH = 512
) (
    input  logic                     clk_i,
    input  logic                     rst_ni,
    input  logic                     cmd_valid_i,
    input  logic [ADDR_WIDTH-1:0]    cmd_addr_i,
    input  logic [DATA_WIDTH-1:0]    cmd_wdata_i,
    input  logic                     cmd_write_i,
    output logic                     cmd_ready_o,
    output logic [DATA_WIDTH-1:0]    rsp_rdata_o,
    output logic                     rsp_valid_o
);

    logic [DATA_WIDTH-1:0] memory [0:(1<<12)-1];
    logic [DATA_WIDTH-1:0] read_data;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            cmd_ready_o <= 1'b1;
            rsp_valid_o <= 1'b0;
            read_data <= '0;
        end else begin
            if (cmd_valid_i && cmd_ready_o) begin
                if (cmd_write_i) begin
                    memory[cmd_addr_i[13:2]] <= cmd_wdata_i;
                    rsp_valid_o <= 1'b0;
                end else begin
                    read_data <= memory[cmd_addr_i[13:2]];
                    rsp_valid_o <= 1'b1;
                end
            end else begin
                rsp_valid_o <= 1'b0;
            end
        end
    end

    assign rsp_rdata_o = read_data;

endmodule

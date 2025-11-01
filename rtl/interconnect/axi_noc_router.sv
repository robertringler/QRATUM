// SPDX-License-Identifier: Apache-2.0
// Lightweight AXI-style NoC router handling fixed topology traffic.

module axi_noc_router #(
    parameter int PORTS = 4,
    parameter int DATA_WIDTH = 256
) (
    input  logic                clk_i,
    input  logic                rst_ni,
    input  logic [PORTS-1:0]    valid_i,
    input  logic [PORTS-1:0][DATA_WIDTH-1:0] data_i,
    output logic [PORTS-1:0]    ready_o,
    output logic [PORTS-1:0]    valid_o,
    output logic [PORTS-1:0][DATA_WIDTH-1:0] data_o,
    input  logic [PORTS-1:0]    ready_i
);

    logic [PORTS-1:0] grant;

    always_comb begin
        grant = '0;
        for (int i = 0; i < PORTS; i++) begin
            if (valid_i[i]) begin
                grant[i] = 1'b1;
            end
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_o <= '0;
            data_o <= '0;
        end else begin
            for (int p = 0; p < PORTS; p++) begin
                valid_o[p] <= grant[p] & ready_i[p];
                data_o[p] <= data_i[p];
                ready_o[p] <= ready_i[p];
            end
        end
    end

endmodule

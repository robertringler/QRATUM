#include "verilated.h"
#include "Vgb10_soc_top.h"

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Vgb10_soc_top top;
    vluint64_t cycle = 0;
    while (!Verilated::gotFinish() && cycle < 10) {
        top.clk_i = cycle & 1;
        if (cycle == 0) {
            top.rst_ni = 0;
        } else if (cycle == 2) {
            top.rst_ni = 1;
        }
        top.eval();
        cycle++;
    }
    return 0;
}

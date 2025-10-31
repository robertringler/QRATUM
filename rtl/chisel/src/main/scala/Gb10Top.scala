// SPDX-License-Identifier: Apache-2.0
package quasim.rtl

import chisel3._
import chisel3.util._

class Gb10TopIO extends Bundle {
  val workloadHint = Input(UInt(64.W))
  val perfCounter = Output(UInt(64.W))
  val tensorResult = Output(UInt(64.W))
}

class Gb10Top extends Module {
  val io = IO(new Gb10TopIO)

  val perfCounter = RegInit(0.U(64.W))
  val tensorResult = RegInit(0.U(64.W))

  val workload = RegNext(io.workloadHint)
  perfCounter := perfCounter + workload(5, 0)
  tensorResult := tensorResult + (workload ^ perfCounter)

  io.perfCounter := perfCounter
  io.tensorResult := tensorResult
}

object Gb10TopDriver extends App {
  (new chisel3.stage.ChiselStage).emitSystemVerilog(new Gb10Top)
}

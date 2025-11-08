import React, { useMemo } from "react";

export function ARCLSystemTracer({ couplingMatrix }) {
  const traceValue = useMemo(() => {
    if (!couplingMatrix || couplingMatrix.length === 0) {
      return 0;
    }

    return couplingMatrix.reduce((acc, row, index) => acc + (row[index] || 0), 0);
  }, [couplingMatrix]);

  return (
    <section className="arcl-system-tracer">
      <header>
        <h3>ARCL Coupling Trace</h3>
      </header>
      <p>Trace Value: {traceValue.toFixed(3)}</p>
      <pre>{JSON.stringify(couplingMatrix, null, 2)}</pre>
    </section>
  );
}

ARCLSystemTracer.defaultProps = {
  couplingMatrix: [
    [1, 0],
    [0, 1],
  ],
};

export default ARCLSystemTracer;

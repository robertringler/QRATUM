import React, { useEffect, useState } from "react";

function createCouplingMatrix(strengths) {
  const matrix = strengths.map((row) => row.slice());
  for (let i = 0; i < matrix.length; i += 1) {
    for (let j = i + 1; j < matrix.length; j += 1) {
      const average = (matrix[i][j] + matrix[j][i]) / 2;
      matrix[i][j] = average;
      matrix[j][i] = average;
    }
  }
  return matrix;
}

export function REVULTRASimulation({ metric, frequency, twist }) {
  const [curvature, setCurvature] = useState(0);

  useEffect(() => {
    const matrix = createCouplingMatrix(metric);
    const diagonal = matrix.reduce((acc, row, index) => acc + (row[index] || 0), 0);
    setCurvature(diagonal * frequency * twist);
  }, [metric, frequency, twist]);

  return (
    <article className="revultra-simulation">
      <h3>REVULTRA Temporal Curvature</h3>
      <p>Frequency: {frequency}</p>
      <p>Twist: {twist}</p>
      <p>Curvature Estimate: {curvature.toFixed(3)}</p>
    </article>
  );
}

REVULTRASimulation.defaultProps = {
  metric: [
    [1, 0],
    [0, 1],
  ],
  frequency: 1,
  twist: 1,
};

export default REVULTRASimulation;

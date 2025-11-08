import React from "react";

export function LSSystemGenerator({ seeds, onGenerate }) {
  const handleClick = () => {
    const payload = seeds.map((seed, index) => ({
      id: `ls-${index}`,
      manifoldWeight: seed * 0.5,
    }));
    onGenerate?.(payload);
  };

  return (
    <div className="ls-system-generator">
      <h3>Quantacosmic L-System Generator</h3>
      <p>
        Seeds: <strong>{seeds.join(", ")}</strong>
      </p>
      <button type="button" onClick={handleClick}>
        Generate Symbolic System
      </button>
    </div>
  );
}

LSSystemGenerator.defaultProps = {
  seeds: [1, 2, 3],
  onGenerate: undefined,
};

export default LSSystemGenerator;

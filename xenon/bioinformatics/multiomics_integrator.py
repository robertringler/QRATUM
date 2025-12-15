"""Multi-omics data integration for XENON.

Provides functionality for:
- Integration of genomics, transcriptomics, proteomics, metabolomics data
- Cross-omics correlation analysis
- Systems biology modeling
- Biomarker discovery
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class OmicsData:
    """Multi-omics data container.
    
    Attributes:
        sample_id: Sample identifier
        genomics: Genomic variants
        transcriptomics: Gene expression levels
        proteomics: Protein abundance
        metabolomics: Metabolite concentrations
        metadata: Sample metadata
    """
    
    sample_id: str
    genomics: Dict[str, any] = field(default_factory=dict)
    transcriptomics: Dict[str, float] = field(default_factory=dict)
    proteomics: Dict[str, float] = field(default_factory=dict)
    metabolomics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def get_feature_vector(self, omics_types: List[str]) -> np.ndarray:
        """Get combined feature vector.
        
        Args:
            omics_types: List of omics types to include
        
        Returns:
            Feature vector
        """
        features = []
        
        if "transcriptomics" in omics_types:
            features.extend(self.transcriptomics.values())
        
        if "proteomics" in omics_types:
            features.extend(self.proteomics.values())
        
        if "metabolomics" in omics_types:
            features.extend(self.metabolomics.values())
        
        return np.array(features)


@dataclass
class Biomarker:
    """Potential biomarker.
    
    Attributes:
        feature_id: Feature identifier (gene, protein, metabolite)
        omics_type: Type of omics (genomics, transcriptomics, etc.)
        effect_size: Effect size (fold-change, coefficient, etc.)
        p_value: Statistical significance
        fdr: False discovery rate
        confidence: Confidence score
    """
    
    feature_id: str
    omics_type: str
    effect_size: float
    p_value: float
    fdr: float = 1.0
    confidence: float = 0.0
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if biomarker is statistically significant.
        
        Args:
            alpha: Significance threshold
        
        Returns:
            True if significant
        """
        return self.fdr < alpha


class MultiOmicsIntegrator:
    """Multi-omics data integration and analysis.
    
    Provides tools for integrating heterogeneous omics data,
    identifying cross-omics patterns, and discovering biomarkers.
    """
    
    def __init__(self):
        """Initialize multi-omics integrator."""
        self._samples: Dict[str, OmicsData] = {}
        self._biomarkers: List[Biomarker] = []
    
    def add_sample(self, sample: OmicsData) -> None:
        """Add an omics sample.
        
        Args:
            sample: OmicsData object
        """
        self._samples[sample.sample_id] = sample
    
    def compute_cross_omics_correlation(
        self,
        omics1: str,
        feature1: str,
        omics2: str,
        feature2: str,
    ) -> Tuple[float, float]:
        """Compute correlation between features across omics layers.
        
        Args:
            omics1: First omics type
            feature1: First feature ID
            omics2: Second omics type
            feature2: Second feature ID
        
        Returns:
            Tuple of (correlation, p_value)
        """
        values1 = []
        values2 = []
        
        for sample in self._samples.values():
            # Get values from omics layers
            val1 = None
            val2 = None
            
            if omics1 == "transcriptomics":
                val1 = sample.transcriptomics.get(feature1)
            elif omics1 == "proteomics":
                val1 = sample.proteomics.get(feature1)
            elif omics1 == "metabolomics":
                val1 = sample.metabolomics.get(feature1)
            
            if omics2 == "transcriptomics":
                val2 = sample.transcriptomics.get(feature2)
            elif omics2 == "proteomics":
                val2 = sample.proteomics.get(feature2)
            elif omics2 == "metabolomics":
                val2 = sample.metabolomics.get(feature2)
            
            if val1 is not None and val2 is not None:
                values1.append(val1)
                values2.append(val2)
        
        if len(values1) < 3:
            return 0.0, 1.0
        
        # Compute Pearson correlation
        corr_matrix = np.corrcoef(values1, values2)
        correlation = float(corr_matrix[0, 1])
        
        # Simplified p-value (Phase 2+ would use proper statistical test)
        n = len(values1)
        t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2 + 1e-10)
        # Approximate p-value
        p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat / np.sqrt(2))))
        
        return correlation, float(p_value)
    
    def identify_biomarkers(
        self,
        group1_samples: List[str],
        group2_samples: List[str],
        omics_types: List[str] = None,
        effect_size_threshold: float = 1.5,
    ) -> List[Biomarker]:
        """Identify potential biomarkers differentiating two groups.
        
        Args:
            group1_samples: Sample IDs in group 1 (e.g., case)
            group2_samples: Sample IDs in group 2 (e.g., control)
            omics_types: Omics types to analyze
            effect_size_threshold: Minimum fold-change
        
        Returns:
            List of potential biomarkers
        """
        if omics_types is None:
            omics_types = ["transcriptomics", "proteomics", "metabolomics"]
        
        biomarkers = []
        
        for omics_type in omics_types:
            # Get all features in this omics layer
            all_features: Set[str] = set()
            for sample in self._samples.values():
                if omics_type == "transcriptomics":
                    all_features.update(sample.transcriptomics.keys())
                elif omics_type == "proteomics":
                    all_features.update(sample.proteomics.keys())
                elif omics_type == "metabolomics":
                    all_features.update(sample.metabolomics.keys())
            
            # Test each feature
            for feature in all_features:
                group1_values = []
                group2_values = []
                
                # Collect values for each group
                for sample_id in group1_samples:
                    sample = self._samples.get(sample_id)
                    if sample:
                        val = self._get_omics_value(sample, omics_type, feature)
                        if val is not None:
                            group1_values.append(val)
                
                for sample_id in group2_samples:
                    sample = self._samples.get(sample_id)
                    if sample:
                        val = self._get_omics_value(sample, omics_type, feature)
                        if val is not None:
                            group2_values.append(val)
                
                if len(group1_values) < 2 or len(group2_values) < 2:
                    continue
                
                # Compute statistics
                mean1 = np.mean(group1_values)
                mean2 = np.mean(group2_values)
                
                # Fold change (log2)
                if mean2 > 0:
                    fold_change = mean1 / mean2
                    log2fc = np.log2(fold_change + 1e-10)
                else:
                    log2fc = 0.0
                
                # T-test (simplified)
                var1 = np.var(group1_values)
                var2 = np.var(group2_values)
                n1 = len(group1_values)
                n2 = len(group2_values)
                
                pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
                t_stat = (mean1 - mean2) / np.sqrt(pooled_var * (1/n1 + 1/n2) + 1e-10)
                
                # Approximate p-value
                p_value = 2 * (1 - 0.5 * (1 + np.tanh(abs(t_stat) / np.sqrt(2))))
                
                if abs(log2fc) >= np.log2(effect_size_threshold):
                    biomarker = Biomarker(
                        feature_id=feature,
                        omics_type=omics_type,
                        effect_size=float(log2fc),
                        p_value=float(p_value),
                    )
                    biomarkers.append(biomarker)
        
        # FDR correction (Benjamini-Hochberg)
        if biomarkers:
            p_values = [b.p_value for b in biomarkers]
            fdr_values = self._benjamini_hochberg(p_values)
            
            for biomarker, fdr in zip(biomarkers, fdr_values):
                biomarker.fdr = fdr
                biomarker.confidence = 1.0 - fdr
        
        self._biomarkers = biomarkers
        return biomarkers
    
    def _get_omics_value(
        self,
        sample: OmicsData,
        omics_type: str,
        feature: str,
    ) -> Optional[float]:
        """Get omics value for a feature.
        
        Args:
            sample: OmicsData object
            omics_type: Type of omics
            feature: Feature ID
        
        Returns:
            Feature value if found
        """
        if omics_type == "transcriptomics":
            return sample.transcriptomics.get(feature)
        elif omics_type == "proteomics":
            return sample.proteomics.get(feature)
        elif omics_type == "metabolomics":
            return sample.metabolomics.get(feature)
        return None
    
    def _benjamini_hochberg(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction.
        
        Args:
            p_values: List of p-values
        
        Returns:
            List of FDR-corrected values
        """
        n = len(p_values)
        if n == 0:
            return []
        
        # Sort p-values with indices
        indexed_pvals = sorted(enumerate(p_values), key=lambda x: x[1])
        
        # Compute FDR
        fdr = [0.0] * n
        for i, (idx, pval) in enumerate(indexed_pvals):
            fdr[idx] = pval * n / (i + 1)
        
        # Enforce monotonicity
        for i in range(n - 1, 0, -1):
            if fdr[i] < fdr[i - 1]:
                fdr[i - 1] = fdr[i]
        
        return [min(1.0, f) for f in fdr]
    
    def perform_pathway_enrichment(
        self,
        biomarker_features: List[str],
        pathways: Dict[str, List[str]],
    ) -> List[Tuple[str, float, float]]:
        """Perform pathway enrichment on biomarkers.
        
        Args:
            biomarker_features: List of biomarker feature IDs
            pathways: Dictionary mapping pathway IDs to feature lists
        
        Returns:
            List of (pathway_id, p_value, enrichment_ratio) tuples
        """
        from scipy.stats import hypergeom
        
        biomarker_set = set(biomarker_features)
        
        # Get all features
        all_features = set()
        for features in pathways.values():
            all_features.update(features)
        
        N = len(all_features)  # Population size
        n = len(biomarker_set)  # Sample size
        
        enrichment_results = []
        
        for pathway_id, pathway_features in pathways.items():
            pathway_set = set(pathway_features)
            
            K = len(pathway_set & all_features)  # Pathway size
            k = len(pathway_set & biomarker_set)  # Overlap
            
            if k == 0:
                continue
            
            # Hypergeometric test
            p_value = float(hypergeom.sf(k - 1, N, K, n))
            
            # Enrichment ratio
            expected = (K / N) * n
            enrichment_ratio = k / expected if expected > 0 else 0.0
            
            enrichment_results.append((pathway_id, p_value, enrichment_ratio))
        
        enrichment_results.sort(key=lambda x: x[1])
        return enrichment_results
    
    def build_integrated_network(
        self,
        correlation_threshold: float = 0.7,
    ) -> Dict[str, any]:
        """Build integrated multi-omics network.
        
        Args:
            correlation_threshold: Minimum correlation for edges
        
        Returns:
            Dictionary with nodes and edges
        """
        nodes = []
        edges = []
        
        # Collect all features from all omics types
        features_by_omics: Dict[str, Set[str]] = {
            "transcriptomics": set(),
            "proteomics": set(),
            "metabolomics": set(),
        }
        
        for sample in self._samples.values():
            features_by_omics["transcriptomics"].update(sample.transcriptomics.keys())
            features_by_omics["proteomics"].update(sample.proteomics.keys())
            features_by_omics["metabolomics"].update(sample.metabolomics.keys())
        
        # Add nodes
        node_id = 0
        feature_to_node: Dict[Tuple[str, str], int] = {}
        
        for omics_type, features in features_by_omics.items():
            for feature in features:
                nodes.append({
                    "id": node_id,
                    "feature_id": feature,
                    "omics_type": omics_type,
                })
                feature_to_node[(omics_type, feature)] = node_id
                node_id += 1
        
        # Compute correlations and add edges
        omics_types = list(features_by_omics.keys())
        
        for i, omics1 in enumerate(omics_types):
            for omics2 in omics_types[i:]:
                for feat1 in features_by_omics[omics1]:
                    for feat2 in features_by_omics[omics2]:
                        if omics1 == omics2 and feat1 == feat2:
                            continue
                        
                        corr, pval = self.compute_cross_omics_correlation(
                            omics1, feat1, omics2, feat2
                        )
                        
                        if abs(corr) >= correlation_threshold and pval < 0.05:
                            source_id = feature_to_node[(omics1, feat1)]
                            target_id = feature_to_node[(omics2, feat2)]
                            
                            edges.append({
                                "source": source_id,
                                "target": target_id,
                                "correlation": corr,
                                "p_value": pval,
                            })
        
        return {
            "nodes": nodes,
            "edges": edges,
        }
    
    def summarize_sample(self, sample_id: str) -> Dict[str, any]:
        """Generate summary statistics for a sample.
        
        Args:
            sample_id: Sample identifier
        
        Returns:
            Dictionary with summary statistics
        """
        sample = self._samples.get(sample_id)
        if not sample:
            return {}
        
        summary = {
            "sample_id": sample_id,
            "num_transcripts": len(sample.transcriptomics),
            "num_proteins": len(sample.proteomics),
            "num_metabolites": len(sample.metabolomics),
            "metadata": sample.metadata,
        }
        
        # Compute basic statistics
        if sample.transcriptomics:
            values = list(sample.transcriptomics.values())
            summary["transcriptomics_mean"] = float(np.mean(values))
            summary["transcriptomics_std"] = float(np.std(values))
        
        if sample.proteomics:
            values = list(sample.proteomics.values())
            summary["proteomics_mean"] = float(np.mean(values))
            summary["proteomics_std"] = float(np.std(values))
        
        if sample.metabolomics:
            values = list(sample.metabolomics.values())
            summary["metabolomics_mean"] = float(np.mean(values))
            summary["metabolomics_std"] = float(np.std(values))
        
        return summary

"""
Chief Research Orchestrator

Coordinates all genealogical research agents and synthesizes results.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agents import (
    ModernRecordsAgent,
    ColonialAmericanAgent,
    EnglishNobilityAgent,
    MedievalRoyalAgent,
    ProofAnalysisAgent,
    HeraldicAgent,
)


class ChiefResearchOrchestrator:
    """
    Chief Research Orchestrator supervising coordinated team of expert agents.
    
    Executes forensic, archival, publication-grade genealogical investigation
    to prove royal and noble ancestry with primary-source verification.
    """
    
    def __init__(self, subject_name: str = "Robert Ringler Jr."):
        """
        Initialize the orchestrator.
        
        Args:
            subject_name: Name of the subject for genealogical investigation
        """
        self.subject_name = subject_name
        self.agents = self._initialize_agents()
        self.execution_start: Optional[datetime] = None
        self.execution_end: Optional[datetime] = None
        self.results: Dict[str, Any] = {}
        
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all research agents."""
        return {
            "agent_1_modern": ModernRecordsAgent(),
            "agent_2_colonial": ColonialAmericanAgent(),
            "agent_3_nobility": EnglishNobilityAgent(),
            "agent_4_medieval": MedievalRoyalAgent(),
            "agent_5_proof": ProofAnalysisAgent(),
            "agent_6_heraldic": HeraldicAgent(),
        }
    
    def execute_research(self, parallel: bool = True) -> Dict[str, Any]:
        """
        Execute complete genealogical research investigation.
        
        Args:
            parallel: If True, execute agents in parallel; if False, sequential
            
        Returns:
            Complete research results and synthesis
        """
        print(f"\n{'='*80}")
        print(f"GENEALOGICAL PROOF INVESTIGATION: {self.subject_name}")
        print(f"{'='*80}\n")
        
        self.execution_start = datetime.now()
        
        # Context for all agents
        context = {
            "subject_name": self.subject_name,
            "target_royal_ancestors": ["King Edward I"],
            "target_noble_houses": ["Howard", "Plantagenet", "Mowbray", "Grey", "Arundell"],
        }
        
        # Execute primary research agents (1-4 and 6)
        print("Phase 1: Executing primary research agents (parallel execution)...")
        primary_agents = [
            "agent_1_modern",
            "agent_2_colonial", 
            "agent_3_nobility",
            "agent_4_medieval",
            "agent_6_heraldic",
        ]
        
        primary_results = self._execute_agents(primary_agents, context, parallel=parallel)
        
        # Execute proof analysis agent (5) with results from other agents
        print("\nPhase 2: Executing proof analysis and validation...")
        context["all_agents_results"] = [primary_results[agent_id] for agent_id in primary_agents]
        
        proof_result = self.agents["agent_5_proof"].execute_research(context)
        primary_results["agent_5_proof"] = proof_result
        
        self.results = primary_results
        self.execution_end = datetime.now()
        
        # Generate final synthesis
        print("\nPhase 3: Generating final synthesis...")
        synthesis = self._generate_synthesis()
        
        print(f"\n{'='*80}")
        print("RESEARCH EXECUTION COMPLETE")
        print(f"{'='*80}\n")
        
        return {
            "subject": self.subject_name,
            "execution_time": str(self.execution_end - self.execution_start),
            "agents_results": self.results,
            "synthesis": synthesis,
        }
    
    def _execute_agents(
        self, agent_ids: List[str], context: Dict[str, Any], parallel: bool = True
    ) -> Dict[str, Any]:
        """Execute specified agents."""
        results = {}
        
        if parallel:
            # Execute agents in parallel
            with ThreadPoolExecutor(max_workers=len(agent_ids)) as executor:
                future_to_agent = {
                    executor.submit(self.agents[agent_id].execute_research, context): agent_id
                    for agent_id in agent_ids
                }
                
                for future in as_completed(future_to_agent):
                    agent_id = future_to_agent[future]
                    try:
                        result = future.result()
                        results[agent_id] = result
                        print(f"  ✓ {agent_id} completed")
                    except Exception as e:
                        print(f"  ✗ {agent_id} failed: {str(e)}")
                        results[agent_id] = {"success": False, "error": str(e)}
        else:
            # Execute agents sequentially
            for agent_id in agent_ids:
                try:
                    result = self.agents[agent_id].execute_research(context)
                    results[agent_id] = result
                    print(f"  ✓ {agent_id} completed")
                except Exception as e:
                    print(f"  ✗ {agent_id} failed: {str(e)}")
                    results[agent_id] = {"success": False, "error": str(e)}
        
        return results
    
    def _generate_synthesis(self) -> Dict[str, Any]:
        """Generate final synthesis of all research."""
        
        # Extract proof determination from Agent 5
        proof_analysis = self.results.get("agent_5_proof", {})
        proof_determination = proof_analysis.get("proof_determination", {})
        
        synthesis = {
            "executive_summary": self._generate_executive_summary(proof_determination),
            "generation_table": self._generate_generation_table(),
            "descent_chart": self._generate_descent_chart(),
            "proof_argument": self._generate_proof_argument(proof_determination),
            "limitations": self._extract_limitations(proof_determination),
            "recommendations": self._compile_recommendations(),
        }
        
        return synthesis
    
    def _generate_executive_summary(self, proof_determination: Dict) -> str:
        """Generate executive summary of research findings."""
        
        overall = proof_determination.get("overall_determination", "UNKNOWN")
        
        summary = f"""
EXECUTIVE SUMMARY: GENEALOGICAL PROOF INVESTIGATION
Subject: {self.subject_name}
Overall Determination: {overall}

This investigation sought to establish proof of royal and noble ancestry for 
{self.subject_name}, specifically tracing descent from King Edward I of England
and related noble houses (Howard, Plantagenet, Mowbray, Grey, Arundell).

PROVEN ELEMENTS:
"""
        
        for element in proof_determination.get("proven_elements", []):
            summary += f"  • {element}\n"
        
        summary += "\nUNPROVEN ELEMENTS:\n"
        for element in proof_determination.get("unproven_elements", []):
            summary += f"  • {element}\n"
        
        summary += f"""
CONCLUSION:
{proof_determination.get("conclusion", "Investigation incomplete.")}

RECOMMENDATION:
{proof_determination.get("recommendation", "Additional research required.")}
"""
        
        return summary
    
    def _generate_generation_table(self) -> List[Dict[str, str]]:
        """Generate generation-by-generation proof table."""
        
        # This would be populated with actual genealogical data
        # For demonstration, showing the structure
        
        table = [
            {
                "generation": "1 (Subject)",
                "person": self.subject_name,
                "dates": "b. [date needed]",
                "relationship": "Subject",
                "evidence": "Birth certificate required",
                "proof_status": "Incomplete"
            },
            {
                "generation": "2 (Parents)",
                "person": "Robert Ringler Sr. (father)",
                "dates": "b. [date needed]",
                "relationship": "Father of subject",
                "evidence": "Birth certificate, marriage certificate needed",
                "proof_status": "Incomplete"
            },
            {
                "generation": "[Colonial Period]",
                "person": "[Connection to colonial families needed]",
                "dates": "1600-1800",
                "relationship": "[Multiple generations]",
                "evidence": "Land records, wills, church registers required",
                "proof_status": "Unproven"
            },
            {
                "generation": "[English Gentry]",
                "person": "[Connection to armigerous families needed]",
                "dates": "1400-1600",
                "relationship": "[Multiple generations]",
                "evidence": "Visitations, wills, IPMs required",
                "proof_status": "Unproven connection to Ringler line"
            },
            {
                "generation": "[Nobility]",
                "person": "Howard, Mowbray, Grey families (documented)",
                "dates": "1300-1600",
                "relationship": "[IF connected through colonial/gentry lines]",
                "evidence": "Complete Peerage, IPMs, charters (PROVEN for noble families themselves)",
                "proof_status": "Noble families proven; connection to subject unproven"
            },
            {
                "generation": "[Royal]",
                "person": "Thomas of Brotherton, Earl of Norfolk",
                "dates": "1300-1338",
                "relationship": "Son of Edward I (PROVEN descent to nobility)",
                "evidence": "Royal charters, chronicles, Complete Peerage",
                "proof_status": "PROVEN to nobility; connection to subject unproven"
            },
            {
                "generation": "[Royal - Target]",
                "person": "Edward I, King of England",
                "dates": "1239-1307",
                "relationship": "Common royal ancestor (PROVEN to nobility)",
                "evidence": "Contemporary chronicles, royal records",
                "proof_status": "PROVEN to nobility; connection to subject unproven"
            },
        ]
        
        return table
    
    def _generate_descent_chart(self) -> str:
        """Generate descent chart from Edward I to subject."""
        
        chart = """
DESCENT CHART: Edward I → [Ringler Family]

Edward I (1239-1307), King of England
    |
    +-- Thomas of Brotherton (1300-1338), Earl of Norfolk
        |
        +-- Margaret of Brotherton (c.1320-1399), suo jure Countess of Norfolk
            |
            +-- [via Segrave] → Elizabeth Segrave
                |
                +-- [via Mowbray] → John Mowbray, 4th Baron Mowbray
                    |
                    +-- Thomas Mowbray (1366-1399), 1st Duke of Norfolk
                        |
                        +-- [Multiple Mowbray generations]
                            |
                            +-- Anne Mowbray (d. 1481) [Mowbray estates and honors]
                                |
                                +-- [via Howard marriage] → Howard family (Dukes of Norfolk, 2nd creation)
                                    |
                                    +-- [PROVEN DESCENT TO HOWARD FAMILY AND RELATED NOBLE HOUSES]
                                    |
                                    +-- [CRITICAL GAP: Connection to English gentry/colonial families]
                                        |
                                        +-- [CRITICAL GAP: Colonial American gentry connection]
                                            |
                                            +-- [CRITICAL GAP: Connection to Ringler surname in America]
                                                |
                                                +-- Robert Ringler Sr. (dates needed)
                                                    |
                                                    +-- Robert Ringler Jr. (SUBJECT)

LEGEND:
  [PROVEN] = Strong documentary evidence
  [CRITICAL GAP] = Requires additional primary source research
  → = Descent through (with marriage or inheritance)

NOTE: The descent from Edward I to English noble families (Howard, Mowbray, etc.)
is WELL-DOCUMENTED with primary sources. The connection from these noble families
to the subject Robert Ringler Jr. requires substantial additional research.
"""
        
        return chart
    
    def _generate_proof_argument(self, proof_determination: Dict) -> str:
        """Generate formal narrative proof argument."""
        
        argument = f"""
FORMAL PROOF ARGUMENT
Subject: {self.subject_name}
Standard: Board for Certification of Genealogists Genealogical Proof Standard (GPS)

I. RESEARCH OBJECTIVE
To determine whether {self.subject_name} descends from King Edward I of England
and related English noble houses through documented, verifiable genealogical connections.

II. GENEALOGICAL PROOF STANDARD ASSESSMENT

A. Reasonably Exhaustive Research:
Status: {proof_determination.get('gps_compliance', {}).get('criterion_1_exhaustive_research', {}).get('status', 'Unknown')}
The research has comprehensively examined:
- Medieval royal and noble records for Plantagenet and related families
- English nobility documentation (Complete Peerage, Heralds' Visitations, etc.)
- Published colonial American gentry research

Critical gaps remain in:
- Colonial American records for Ringler surname
- Vital records for modern generations
- Connection between colonial and English gentry lines

B. Complete and Accurate Citations:
Status: {proof_determination.get('gps_compliance', {}).get('criterion_2_citations', {}).get('status', 'Unknown')}
Noble and royal sources are properly cited with authoritative references.
Modern and colonial sources require additional specific citations.

C. Thorough Analysis and Correlation:
Status: {proof_determination.get('gps_compliance', {}).get('criterion_3_analysis_correlation', {}).get('status', 'Unknown')}
Strong correlation exists for medieval royal to nobility descents.
Weak correlation between modern subject and historical noble lines.

D. Resolution of Conflicting Evidence:
Status: {proof_determination.get('gps_compliance', {}).get('criterion_4_conflict_resolution', {}).get('status', 'Unknown')}
No major conflicts in documented noble descents.
Potential conflicts in surname variations not yet addressed.

E. Soundly Written Conclusion:
Based on current evidence, the genealogical proof is INCOMPLETE.

III. PROVEN DESCENTS

A. Edward I to Thomas of Brotherton:
Thomas of Brotherton (1300-1338) was the legitimate son of King Edward I
by his second wife, Margaret of France. This is established by:
- Contemporary chronicles
- Royal charters and grants
- Complete Peerage documentation
Proof quality: STRONG PRIMARY SOURCES

B. Thomas of Brotherton to Mowbray Family:
Descent through Margaret of Brotherton to the Mowbray Dukes of Norfolk
is well-documented through:
- Inquisitions Post Mortem
- Heralds' records
- Estate succession documents
Proof quality: STRONG PRIMARY SOURCES

C. Mowbray to Howard Family:
The Howard family inherited Mowbray estates and the Duke of Norfolk title
through marriage and legal succession, documented in:
- Complete Peerage
- Patent Rolls
- Court records
Proof quality: STRONG PRIMARY SOURCES

IV. UNPROVEN CONNECTIONS

A. Howard/Noble Families to Colonial American Families:
Status: UNPROVEN for Ringler line specifically
While many colonial American families demonstrate proven English noble origins,
no specific connection has been established for the Ringler surname.

B. Colonial American Families to Modern Ringler Family:
Status: UNPROVEN
Comprehensive colonial records research for Ringler surname required.

C. Modern Generations:
Status: PARTIAL
Complete vital records documentation required for all generations 1800-present.

V. CONCLUSION

The genealogical evidence demonstrates STRONG PROOF of descent from
King Edward I to English noble families (Howard, Mowbray, Grey, etc.).

However, the specific connection from these documented noble lines to
{self.subject_name} remains UNPROVEN without additional primary source research.

Current evidence: INSUFFICIENT to meet Genealogical Proof Standard for
claimed royal/noble descent of {self.subject_name}.

Recommendation: MAJOR ADDITIONAL RESEARCH REQUIRED in colonial American
records and modern vital records to establish or refute the claimed descent.
"""
        
        return argument
    
    def _extract_limitations(self, proof_determination: Dict) -> List[str]:
        """Extract research limitations."""
        
        limitations = [
            "Incomplete vital records for modern generations (1800-present)",
            "No documented connection between Ringler surname and colonial gentry families",
            "Insufficient colonial American records research for Ringler surname",
            "Maternal line surnames and documentation gaps",
            "Surname variations in historical records not fully investigated",
            "DNA evidence not utilized to support or refute genealogical connections",
        ]
        
        return limitations
    
    def _compile_recommendations(self) -> List[str]:
        """Compile recommendations for future research."""
        
        recommendations = [
            "Conduct exhaustive search of colonial Maryland, Virginia, and Pennsylvania records for Ringler surname and variations",
            "Obtain certified copies of all vital records (birth, marriage, death) for generations 1800-present",
            "Investigate maternal lines for all generations to identify potential noble connections through female ancestry",
            "Consult specialized colonial American genealogist for surname research",
            "Search naturalization and immigration records for Ringler family arrival in America",
            "Examine church registers for all relevant parishes and denominations",
            "Research land records, wills, and probate records in colonial jurisdictions",
            "Consider autosomal DNA testing to identify potential connections to documented noble-descent families",
            "Investigate all surname variations (Ringler, Ringer, Rengler, etc.) in historical records",
            "Consult with Board-Certified Genealogist for professional evaluation",
        ]
        
        return recommendations
    
    def export_results(self, filepath: str, format: str = "json") -> None:
        """
        Export results to file.
        
        Args:
            filepath: Path to export file
            format: Export format ('json', 'txt', or 'html')
        """
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
        elif format == "txt":
            with open(filepath, 'w') as f:
                f.write(self._format_text_report())
        elif format == "html":
            with open(filepath, 'w') as f:
                f.write(self._format_html_report())
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_text_report(self) -> str:
        """Format results as text report."""
        synthesis = self.results.get("synthesis", {}) if "synthesis" in self.results else self._generate_synthesis()
        
        report = f"""
{'='*80}
GENEALOGICAL PROOF INVESTIGATION REPORT
{'='*80}

{synthesis.get('executive_summary', '')}

{'='*80}
DESCENT CHART
{'='*80}

{synthesis.get('descent_chart', '')}

{'='*80}
FORMAL PROOF ARGUMENT
{'='*80}

{synthesis.get('proof_argument', '')}

{'='*80}
LIMITATIONS
{'='*80}

"""
        for limitation in synthesis.get('limitations', []):
            report += f"  • {limitation}\n"
        
        report += f"""
{'='*80}
RECOMMENDATIONS FOR FUTURE RESEARCH
{'='*80}

"""
        for i, rec in enumerate(synthesis.get('recommendations', []), 1):
            report += f"{i}. {rec}\n"
        
        return report
    
    def _format_html_report(self) -> str:
        """Format results as HTML report."""
        synthesis = self.results.get("synthesis", {}) if "synthesis" in self.results else self._generate_synthesis()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Genealogical Proof Investigation: {self.subject_name}</title>
    <style>
        body {{ font-family: Georgia, serif; max-width: 900px; margin: 40px auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }}
        .summary {{ background: #ecf0f1; padding: 20px; border-left: 4px solid #3498db; }}
        .chart {{ background: #f9f9f9; padding: 15px; font-family: monospace; white-space: pre; overflow-x: auto; }}
        .limitation, .recommendation {{ margin: 10px 0; padding-left: 20px; }}
        .proven {{ color: #27ae60; font-weight: bold; }}
        .unproven {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Genealogical Proof Investigation Report</h1>
    <p><strong>Subject:</strong> {self.subject_name}</p>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <pre>{synthesis.get('executive_summary', '')}</pre>
    </div>
    
    <h2>Descent Chart</h2>
    <div class="chart">{synthesis.get('descent_chart', '')}</div>
    
    <h2>Formal Proof Argument</h2>
    <pre>{synthesis.get('proof_argument', '')}</pre>
    
    <h2>Research Limitations</h2>
    <ul>
"""
        
        for limitation in synthesis.get('limitations', []):
            html += f"        <li class='limitation'>{limitation}</li>\n"
        
        html += """    </ul>
    
    <h2>Recommendations for Future Research</h2>
    <ol>
"""
        
        for rec in synthesis.get('recommendations', []):
            html += f"        <li class='recommendation'>{rec}</li>\n"
        
        html += """    </ol>
</body>
</html>
"""
        
        return html

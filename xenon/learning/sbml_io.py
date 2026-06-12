"""SBML import/export for XENON mechanisms (Phase 2: ecosystem interoperability).

Round-trips a ``BioMechanism`` (species = molecular states, reactions = mass-action
transitions ``source -> target`` with rate law ``k * source``) to and from SBML
Level 3, so XENON can ingest curated community models (e.g. BioModels) and its
mechanisms can be simulated by standard SBML engines (libRoadRunner, COPASI,
Tellurium). Requires ``python-libsbml`` (optional dependency).
"""

from __future__ import annotations

from typing import Optional

from ..core.mechanism import BioMechanism, MolecularState, Transition

try:
    import libsbml  # type: ignore

    _HAS_LIBSBML = True
except ImportError:  # pragma: no cover - optional dependency
    libsbml = None  # type: ignore
    _HAS_LIBSBML = False


def _require_libsbml() -> None:
    if not _HAS_LIBSBML:
        raise ImportError(
            "python-libsbml is required for SBML I/O; install with 'pip install python-libsbml'"
        )


def _check(code: int, what: str) -> None:
    """Raise if a libSBML setter returned a non-OK status."""
    if code is not None and code != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise RuntimeError(f"libSBML error setting {what}: status {code}")


def mechanism_to_sbml(mechanism: BioMechanism, compartment_size: float = 1.0) -> str:
    """Serialize a mechanism to an SBML Level 3 Version 2 document string."""
    _require_libsbml()
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId((mechanism.name or "mechanism").replace(" ", "_"))

    comp = model.createCompartment()
    comp.setId("cell")
    comp.setConstant(True)
    comp.setSize(float(compartment_size))
    comp.setSpatialDimensions(3)

    for state in mechanism.states:
        sp = model.createSpecies()
        sp.setId(state.name)
        sp.setCompartment("cell")
        sp.setConstant(False)
        sp.setBoundaryCondition(False)
        sp.setHasOnlySubstanceUnits(False)
        sp.setInitialConcentration(
            float(state.concentration) if state.concentration is not None else 0.0
        )

    for i, trans in enumerate(mechanism.transitions):
        rxn = model.createReaction()
        rxn.setId(f"r{i}")
        rxn.setReversible(False)

        reactant = rxn.createReactant()
        reactant.setSpecies(trans.source)
        reactant.setStoichiometry(1.0)
        reactant.setConstant(True)

        product = rxn.createProduct()
        product.setSpecies(trans.target)
        product.setStoichiometry(1.0)
        product.setConstant(True)

        klaw = rxn.createKineticLaw()
        math_ast = libsbml.parseL3Formula(f"k_{i} * {trans.source}")
        klaw.setMath(math_ast)
        param = klaw.createLocalParameter()
        param.setId(f"k_{i}")
        param.setValue(float(trans.rate_constant))

    return libsbml.writeSBMLToString(doc)


def sbml_to_mechanism(sbml_string: str) -> BioMechanism:
    """Parse an SBML string into a BioMechanism (mass-action unimolecular)."""
    _require_libsbml()
    doc = libsbml.readSBMLFromString(sbml_string)
    if doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR) > 0:
        raise ValueError(f"Invalid SBML: {doc.getErrorLog().toString()}")
    model = doc.getModel()
    if model is None:
        raise ValueError("SBML document contains no model")

    mech = BioMechanism(model.getId() or "mechanism")
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        conc: Optional[float] = (
            float(sp.getInitialConcentration()) if sp.isSetInitialConcentration() else None
        )
        mech.add_state(MolecularState(name=sp.getId(), molecule=sp.getId(), concentration=conc))

    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        if rxn.getNumReactants() == 0 or rxn.getNumProducts() == 0:
            continue
        source = rxn.getReactant(0).getSpecies()
        target = rxn.getProduct(0).getSpecies()
        rate = 1.0
        klaw = rxn.getKineticLaw()
        if klaw is not None and klaw.getNumLocalParameters() > 0:
            rate = float(klaw.getLocalParameter(0).getValue())
        mech.add_transition(Transition(source=source, target=target, rate_constant=rate))

    return mech


__all__ = ["mechanism_to_sbml", "sbml_to_mechanism"]

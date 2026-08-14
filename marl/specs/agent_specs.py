"""Names and ordered fields shared by implementation, tests, and documentation."""

IDC_AGENT = "idc"
BESS_AGENT = "bess"
AGENTS = (IDC_AGENT, BESS_AGENT)

ACTION_PADDING_STRATEGY = "padding-v1-effective-mask"
INPUT_SEMANTICS_VERSION = "idc25-normalized-supplemental-v1"
SUPPLEMENTAL_NORMALIZATION_VERSION = "physical-reference-v1"
PADDED_ACTION_DIMS = {
    IDC_AGENT: 22,
    BESS_AGENT: 22,
}
EFFECTIVE_ACTION_DIMS = {
    IDC_AGENT: 22,
    BESS_AGENT: 1,
}


def effective_action_mask(agent: str) -> tuple[float, ...]:
    """Return the immutable padded-action mask defined by the agent contract."""
    if agent not in AGENTS:
        raise KeyError(f"Unknown agent {agent!r}; expected one of {AGENTS!r}.")
    padded_dim = PADDED_ACTION_DIMS[agent]
    effective_dim = EFFECTIVE_ACTION_DIMS[agent]
    return (1.0,) * effective_dim + (0.0,) * (padded_dim - effective_dim)

SUPPLEMENTAL_FIELDS = (
    "bess_soc",
    "bess_energy_kWh",
    "P_IDC_kW",
    "P_grid_kW",
    "bess_charge_power_kW",
    "bess_discharge_power_kW",
)

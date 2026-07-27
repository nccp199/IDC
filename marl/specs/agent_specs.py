"""Names and ordered fields shared by implementation, tests, and documentation."""

IDC_AGENT = "idc"
BESS_AGENT = "bess"
AGENTS = (IDC_AGENT, BESS_AGENT)

SUPPLEMENTAL_FIELDS = (
    "bess_soc",
    "bess_energy_kWh",
    "P_IDC_kW",
    "P_grid_kW",
    "bess_charge_power_kW",
    "bess_discharge_power_kW",
)

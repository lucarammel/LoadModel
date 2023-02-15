from pathlib import Path
import pandas as pd


def process_scenarios_anomalies(data_root: Path, etp_scenario: str, year: int):
    """

    Args:
        data_root:
        etp_scenario: 'SDS' for RCP2.6, 'STEPS' for RCP4.6
        year:

    Returns:

    """
    rcp_scenario = {'SDS': '26',
                    'STEPS': '45'}
    df = pd.read_csv(data_root.joinpath(f'raw/scenarios_anomalies/ETP{rcp_scenario[etp_scenario]}_{year}_antemp.csv'))

    # reshape data frame
    df = df.rename(columns={'CountryOUT': 'country'}).set_index('country').rename(lambda s: int(s[5:]), axis=1)
    df = df.reset_index().melt(id_vars='country', var_name='month', value_name='temperature_anomaly')

    return df

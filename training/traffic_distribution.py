import pandas as pd
import json

csv_files = [
    'cicids2017/Monday-WorkingHours.csv',
    'cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv',
    'cicids2017/Wednesday-workingHours.pcap_ISCX.csv',
    'cicids2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'cicids2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'cicids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv'
]
data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
data.columns = data.columns.str.strip()

print("Available columns:", data.columns)

label_column = 'Label'

if label_column not in data.columns:
    raise KeyError(f"The specified label column '{label_column}' was not found in the dataset.")


traffic_distribution = data[label_column].value_counts(normalize=True) * 100
traffic_distribution.to_json('traffic_distribution.json', indent=4)
traffic_distribution.to_csv('traffic_distribution.csv', header=['Percentage'])
latex_table = traffic_distribution.to_latex(index=True, header=['Percentage'])
with open('traffic_distribution_table.tex', 'w') as f:
    f.write(latex_table)

print("Traffic distribution saved to 'traffic_distribution.json', 'traffic_distribution.csv', and LaTeX table 'traffic_distribution_table.tex'.")

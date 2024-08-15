import os
import json


directory = './'
output_file = 'consolidated_results.txt'

with open(output_file, 'w') as outfile:
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as json_file:
                content = json.load(json_file)
                outfile.write(f'Filename: {filename}\n')
                json.dump(content, outfile, indent=4)
                outfile.write('\n\n')

print(f'CjSON files have been written to {output_file}.')

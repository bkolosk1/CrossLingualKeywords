import json
import os
for lang in ['english','estonian','latvian','slovenian','croatian']:
    for split in ['_valid.json', '_test.json']:
        with open(f"./{lang}/{lang}{split}", 'r') as f:
            for line in f:
                jsoned = json.loads(line)
                if not 'lang' in jsoned:
                    print(split)


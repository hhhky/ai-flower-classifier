python -c "
import csv

# 读取 credentials
with open(r'C:\Users\14168\Downloads\credentials.csv') as f:
    reader = csv.DictReader(f)
    row = next(reader)
    ak = row['Access Key Id']
    sk = row['Secret Access Key']

# obsutil INI 格式
config = f\"\"\"[config]
endpoint=obs.cn-north-4.myhuaweicloud.com
ak={ak}
sk={sk}
\"\"\"

import os
config_path = os.path.expanduser('~/.obsutilconfig')
with open(config_path, 'w') as f:
    f.write(config)
print(f'配置已写入 (INI 格式)')
"
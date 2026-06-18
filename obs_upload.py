# -*- coding: utf-8 -*-
"""上传 train.py 到 OBS"""
import sys, os, csv, hashlib, hmac, datetime, requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\14168\Downloads\credentials.csv') as f:
    row = next(csv.DictReader(f))
    AK = row['Access Key Id']
    SK = row['Secret Access Key']

REGION = "cn-north-4"
BUCKET = "flower-classify-modelarts"
ENDPOINT = f"https://obs.{REGION}.myhuaweicloud.com"

def sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

def get_signature_key(secret_key, date, region, service):
    k_date = sign(('AWS4' + secret_key).encode('utf-8'), date)
    k_region = sign(k_date, region)
    k_service = sign(k_region, service)
    return sign(k_service, 'aws4_request')

def obs_request(method, path, headers=None, data=None):
    t = datetime.datetime.utcnow()
    amzdate = t.strftime('%Y%m%dT%H%M%SZ')
    datestamp = t.strftime('%Y%m%d')
    url = f"{ENDPOINT}{path}"
    host = f"obs.{REGION}.myhuaweicloud.com"

    req_headers = {
        'Host': host,
        'x-amz-date': amzdate,
        'x-amz-content-sha256': hashlib.sha256(data or b'').hexdigest(),
    }
    if headers:
        req_headers.update(headers)

    signed_headers = ';'.join(sorted(req_headers.keys()))
    canonical_uri = path
    canonical_headers = ''.join(f"{k}:{req_headers[k]}\n" for k in sorted(req_headers.keys()))
    payload_hash = req_headers['x-amz-content-sha256']
    canonical_request = f"{method}\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    algorithm = 'AWS4-HMAC-SHA256'
    credential_scope = f"{datestamp}/{REGION}/s3/aws4_request"
    string_to_sign = f"{algorithm}\n{amzdate}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
    signing_key = get_signature_key(SK, datestamp, REGION, 's3')
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
    authorization = f"{algorithm} Credential={AK}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
    req_headers['Authorization'] = authorization
    return requests.request(method, url, headers=req_headers, data=data)

# 上传 train_modelarts.py
print("上传 train_modelarts.py ...")
train_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_modelarts.py")
with open(train_py, 'rb') as f:
    content = f.read()

resp = obs_request('PUT', f'/{BUCKET}/train_modelarts.py', data=content)
print(f"  train_modelarts.py: HTTP {resp.status_code} {'✅' if resp.status_code == 200 else resp.text[:100]}")

# 验证可访问
resp = obs_request('HEAD', f'/{BUCKET}/train_modelarts.py')
print(f"  验证: HTTP {resp.status_code}, Size: {resp.headers.get('Content-Length', '?')} bytes")

print(f"\nOBS 路径: obs://{BUCKET}/train_modelarts.py")
print("完成！")

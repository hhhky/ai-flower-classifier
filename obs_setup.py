# -*- coding: utf-8 -*-
"""华为云 OBS 操作脚本 — 纯 Python 实现，只依赖 requests"""
import sys, os, csv, hashlib, hmac, base64, datetime
from xml.etree import ElementTree as ET
import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── 读取凭证 ──
with open(r'C:\Users\14168\Downloads\credentials.csv') as f:
    row = next(csv.DictReader(f))
    AK = row['Access Key Id']
    SK = row['Secret Access Key']

REGION = "cn-north-4"
BUCKET = "flower-classify-modelarts"
# OBS endpoint (S3 兼容)
ENDPOINT = f"https://obs.{REGION}.myhuaweicloud.com"

# ── AWS SigV4 签名 ──
def sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

def get_signature_key(secret_key, date, region, service):
    k_date = sign(('AWS4' + secret_key).encode('utf-8'), date)
    k_region = sign(k_date, region)
    k_service = sign(k_region, service)
    return sign(k_service, 'aws4_request')

def obs_request(method, path, headers=None, data=None):
    """发送带 SigV4 签名的 OBS 请求"""
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

    # 排序
    signed_headers = ';'.join(sorted(req_headers.keys()))

    # Canonical request
    canonical_uri = path
    canonical_querystring = ''
    canonical_headers = ''.join(f"{k}:{req_headers[k]}\n" for k in sorted(req_headers.keys()))
    payload_hash = req_headers['x-amz-content-sha256']

    canonical_request = f"{method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"

    algorithm = 'AWS4-HMAC-SHA256'
    credential_scope = f"{datestamp}/{REGION}/s3/aws4_request"
    string_to_sign = f"{algorithm}\n{amzdate}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"

    signing_key = get_signature_key(SK, datestamp, REGION, 's3')
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

    authorization = f"{algorithm} Credential={AK}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
    req_headers['Authorization'] = authorization

    return requests.request(method, url, headers=req_headers, data=data)

# ── 1. 创建桶 ──
print("=" * 50)
print(f"创建 OBS 桶: {BUCKET}")
print("=" * 50)
# 需要带上区域约束
location_xml = f'<CreateBucketConfiguration><LocationConstraint>{REGION}</LocationConstraint></CreateBucketConfiguration>'
headers = {'Content-Type': 'application/xml'}
resp = obs_request('PUT', f'/{BUCKET}/', headers=headers, data=location_xml.encode('utf-8'))
if resp.status_code in (200, 409):
    print(f"✅ 桶就绪 (HTTP {resp.status_code})")
else:
    print(f"❌ 创建失败: {resp.status_code} {resp.text}")
    sys.exit(1)

print("\n环境准备完成！下一步：上传数据集")

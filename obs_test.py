# -*- coding: utf-8 -*-
"""测试 OBS 连接 — 先列桶看凭证是否有效"""
import sys, os, csv, hashlib, hmac, datetime
import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\14168\Downloads\credentials.csv') as f:
    row = next(csv.DictReader(f))
    AK = row['Access Key Id']
    SK = row['Secret Access Key']

REGION = "cn-north-4"
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

# 1. 列出桶
print("1) 列出 OBS 桶...")
resp = obs_request('GET', '/')
print(f"   HTTP {resp.status_code}")
if resp.status_code == 200:
    root = ET.fromstring(resp.text)
    buckets = root.findall('.//{http://s3.amazonaws.com/doc/2006-03-01/}Bucket')
    for b in buckets:
        name = b.find('{http://s3.amazonaws.com/doc/2006-03-01/}Name')
        if name is not None:
            print(f"   📦 {name.text}")
    if not buckets:
        print("   (空 — 没有任何桶)")
elif resp.status_code == 403:
    print(f"   ❌ 403 Forbidden: {ET.fromstring(resp.text).find('.//{http://s3.amazonaws.com/doc/2006-03-01/}Message').text if 'Message' in resp.text else resp.text[:200]}")

# 2. 试试用 IAM API 验证身份
print("\n2) 测试 IAM 身份验证...")
iam_resp = requests.post(
    "https://iam.cn-north-4.myhuaweicloud.com/v3/auth/tokens",
    json={
        "auth": {
            "identity": {
                "methods": ["password"],
                "password": {"user": {"name": "REPLACE", "password": "REPLACE"}}
            }
        }
    },
    headers={"Content-Type": "application/json"}
)
print(f"   HTTP {iam_resp.status_code} (需要用户名/密码，AK/SK 不走这个接口)")

print(f"\n诊断: 账号类型 — IAM 子用户: hid_h-7_w_t432wzenk")
print(f"可能原因:")
print(f"  1) 主账号未给此子用户开放 OBS 权限")
print(f"  2) 子用户需要在控制台 IAM 中绑定 OBS 策略")
print(f"\n请在浏览器中检查:")
print(f"  华为云控制台 → 统一身份认证 → 用户 → hid_h-7_w_t432wzenk")
print(f"  → 授权 → 添加 'OBS Administrator' 或 'OBS Buckets Viewer'")

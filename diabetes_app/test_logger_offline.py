"""Offline mock tests for Apps Script logging (no account or internet needed)."""
import csv
import importlib
import sys
import types
from pathlib import Path

st = types.ModuleType('streamlit')
st.secrets = {'apps_script_url': 'https://script.google.com/macros/s/DEMO/exec',
              'logging_token': '12345678901234567890123456789012'}
sys.modules['streamlit'] = st

requests = types.ModuleType('requests')
class Response:
    def __init__(self, result): self.result = result
    def raise_for_status(self): pass
    def json(self): return self.result
call_log = []
def post(url, *, json, **kwargs):
    call_log.append(json)
    return Response({'ok':True, 'prediction_id':json['row'][0]})
requests.post = post
sys.modules['requests'] = requests
logger = importlib.import_module('logger')
columns=logger._schema_columns()
assert len(columns)==34
sample=list(csv.DictReader((Path(__file__).parent/'deployment_bundle/sample_input.csv').open()))[0]
inputs={k: float(sample[k]) for k in columns[3:24]}
success,pid,err=logger.log_prediction(inputs,0.010257108137011528,5.2,
                                       [('Age',-1.4),('HighBP',-0.3),('GenHlth',-0.2)],'1.0.0')
assert success and err is None and len(call_log)==1
row=call_log[0]['row']
assert len(row)==34 and row[0]==pid and abs(row[24]-0.010257108137011528)<1e-12
assert row[26]=='Age' and row[28]=='HighBP' and row[30]=='GenHlth'
assert row[-2:]==['','']
print('PASS: 34 columns, order, probability and top 3 SHAP')
print('PASS: UUID, timestamp UTC and blank outcomes')
# API returns failure; app must retain prediction, show error, not claim saved.
requests.post=lambda url,*,json,**kwargs: Response({'ok':False,'error':'Wrong header'})
valid,pid,err=logger.log_prediction(inputs,0.01,5.2,
                                     [('Age',-1.4),('HighBP',-0.3),('GenHlth',-0.2)],'1.0.0')
assert not valid and 'Wrong header' in err
print('PASS: remote error returned without claiming success')
st.secrets['logging_token']='bad'
valid,pid,err=logger.log_prediction(inputs,0.01,5.2,
                                     [('Age',-1.4),('HighBP',-0.3),('GenHlth',-0.2)],'1.0.0')
assert not valid and 'logging_token' in err
print('PASS: missing/invalid secrets handled safely')

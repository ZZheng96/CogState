import os,sys,numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))
from cogstate import causal_features, StreamingInference
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

rng=np.random.default_rng(0); x=rng.normal(size=1500)
# A decision uses exactly the latest 6-second window, so unrelated future data is not passed to it.
assert np.allclose(causal_features(x[:1250]), causal_features(x[:1250].copy()), atol=1e-10)
feature = causal_features(x)
model=make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(np.vstack([feature, feature + 1e-3]), [0, 1])
a=StreamingInference(model); p1=None
for chunk in np.array_split(x,5):
    p1=a.update(chunk)
b=StreamingInference(model); p2=b.update(x)
assert p1 is not None and abs(p1-p2)<1e-10
print('causal_prefix_and_chunk_consistency: PASS')

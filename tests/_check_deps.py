"""Quick dependency check script."""
import sys
print(f"Python {sys.version}")

deps = [
    ('numpy', 'np'),
    ('scipy', 'sp'),
    ('pandas', 'pd'),
    ('yaml', 'yaml'),
    ('sklearn', 'sklearn'),
    ('xgboost', 'xgb'),
    ('matplotlib', 'mpl'),
    ('seaborn', 'sns'),
    ('tqdm', 'tqdm'),
    ('torch', 'torch'),
    ('transformers', 'transformers'),
    ('faiss', 'faiss'),
]
missing = []
for name, short in deps:
    try:
        m = __import__(short)
        ver = getattr(m, '__version__', '?')
        print(f"  [OK] {name}=={ver}")
    except ImportError:
        print(f"  [MISSING] {name}")
        missing.append(name)

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
else:
    print("\nAll core packages installed!")

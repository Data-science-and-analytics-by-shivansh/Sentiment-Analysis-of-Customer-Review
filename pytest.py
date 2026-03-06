# pytest.ini - Pytest configuration file

[pytest]
# Test discovery patterns
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Paths
testpaths = tests

# Coverage options
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=sentiment_analysis
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=70

# Markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    smoke: marks tests as smoke tests

# Warnings
filterwarnings =
    error
    ignore::UserWarning
    ignore::DeprecationWarning

# Minimum Python version
minversion = 3.8

# Logging
log_cli = true
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)8s] %(message)s
log_cli_date_format = %Y-%m-%d %H:%M:%S

# Timeout (prevent hanging tests)
timeout = 300

# Anomaly Detection in Periodic Streams



## Requirements
- Python 3.10+     

##
```bash
pip install -r requirements.txt
python main.py
```


## Limitation
- Univariate
- Only supports sinusoidal patterns for stream generation.
- Predictors do not benifit from the seasonal nature of the data.


## Future Work
- Support multivariate data
- Add more complex generation patterns
- Use a low pass filter to remove the noise
- Add better predictors that are more robust to the seasonal nature of the data (i.e Holt Winters, SH-ESD)
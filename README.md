# Multi-Sensor Vibration Analysis

Critical system failures occurred between December 9-31 2024, requiring predictive fault detection and localization capabilites. A multi-stage, sensor-aware AI system can effectively address challenges across a 3-stage drive train system. Awareness of SME-relevant sensor metrics will provide efficient scaffolding from an AI system. An LLM-as-evaluator approach is best addapted for the sparse, unlabeled training environment.

## Data
We were provided sensor data for 7 different sensors on an agitator machine part. This was raw waveform data recorded every 12 hours containing x, y, z vibration data along with temperature, humidity, timestamp, and sample frequency. This data was collected twice a day for a 2 second period each time. EDA found that the issue started around the 10th or 11th, lasted for several days, and then the asset was sent for maintenance.

## Data Pipeline

Raw waveform data in  JSON format was converted to a dataframe with the x, y, z vibrations at a timestamp along with temperature, humidity, and sample frequency. FFT and RMS transformations were performed on the raw waveform data. For FFT, the top 10 amplitude peaks per sensor and rolling 5 day average were found. for RMS, the current and last 3 day's values were found. The LLM could make a judgment if the current amplitude is significantly different than its 5 day average and the RMS is significantly different from the last 3 days.

## LLM Structure

Vertex AI was used to invoke the Gemini-2.0-flash LLM. The system prompt contained agitator context and information on the input data (user prompt) format. Fault detection status, confidence %, fault classification, frequency domain analysis, and root cause analysis were required in the output. The user prompt contained sensor information, RMS trend analysis, and FFT frequency peaks at a particular timestamp. Combined with the user prompt, the output would contain the sensors that are present at that timestamp - might not be 7 reports. Prompt tuning to make sure faults weren't being unnecessarily detected with confidence calibration thresholds + temperature increase for more nuanced answers. Wanted to make sure the model doesn't make any strong claims unnecessarily.

## Streamlit Dashboard

A dashboard was created to present all findings. Users can toggle specific timestamps and sensors. The FFT analysis (frequency domain analysis) tab gives information on x, y, z FFT peaks along with maximum peaks at critical gear mesh frequencies. The RMS (overall vibration energy) tab gives RMS values by axis and RMS evolution over the two second period data is collected by the window size. The time series analysis tab gives information about the raw waveform data (vibration time series and signal statistics). The AI fault analysis tab gives a condensed system summary giving information on what is going on with all sensors present at a time stamp and diagnosis steps. It also gives a table with each sensor present, which fault is detected in the sensor, and with what confidence level a fault has been detected.
